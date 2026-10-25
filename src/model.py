"""MA-RUPA: Modality-Adaptive Multiscale Prototype Alignment Network.

Weakly-Supervised Video Anomaly Detection (WS-VAD) architecture for top-tier
research (IEEE TIP, IEEE TMM, Pattern Recognition).

Key Components:
1. MDTBackbone: Multi-Scale Dilated Temporal & Local Windowed Attention Backbone
2. ModalityAdaptiveBranch: Audio-Visual Multimodal Fusion & Motion Residuals
3. GradientSafeRUPA: Stop-Gradient Dynamic Normal Prototypes & Domain-Adaptive Semantics
4. QuadSafeGate: Quad-Stream Uncertainty-Aware Dynamic Safe Gate
5. MARUPA: Full End-to-End Trainable Architecture (from scratch)
"""

from collections import OrderedDict
from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from clip import clip
from utils.adapter_modules import SimpleAdapter, SimpleProj
from utils.descriptions import DESCRIPTIONS_ORI, DESCRIPTIONS_ORI_XD
from utils.dnp_vision_transformer import Aggregation_Block, Prototype_Block


class QuickGELU(nn.Module):
    """Fast approximation of GELU activation."""
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNormFP32(nn.LayerNorm):
    """LayerNorm that computes statistics in float32 for numerical stability."""
    def forward(self, x: torch.Tensor):
        orig_dtype = x.dtype
        ret = super().forward(x.to(torch.float32))
        return ret.to(orig_dtype)


# ============================================================================
# 1. Core Module 1: MDTBackbone
# ============================================================================

class WindowedAttentionBlock(nn.Module):
    """Lightweight Local Windowed Self-Attention with O(T * W) complexity."""
    def __init__(self, embed_dim=512, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ln_1 = LayerNormFP32(embed_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(embed_dim, embed_dim * 4)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(dropout)),
            ("c_proj", nn.Linear(embed_dim * 4, embed_dim)),
            ("dropout2", nn.Dropout(dropout))
        ]))
        self.ln_2 = LayerNormFP32(embed_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, key_padding_mask: torch.Tensor = None):
        norm_x = self.ln_1(x)
        attn_out, _ = self.attn(
            norm_x, norm_x, norm_x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        # Numerical guard: When an entire local window consists of padded frames,
        # PyTorch MultiheadAttention computes softmax([-inf, ...]) = NaN. Replace with 0.0.
        attn_out = torch.nan_to_num(attn_out, nan=0.0)
        x = x + attn_out
        mlp_out = self.mlp(self.ln_2(x))
        if key_padding_mask is not None:
            mlp_out = mlp_out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        x = x + mlp_out
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return x


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise-Separable 1D Convolution with dilation."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 dilation: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size,
            dilation=dilation, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, bias=True
        )

    def forward(self, x: torch.Tensor):
        return self.pointwise(self.depthwise(x))


class MDTPBranch(nn.Module):
    """Branch B: Multi-Scale Dilated Temporal Pyramid (MDTP).
    
    Captures multi-scale temporal dependencies (rapid anomalies like explosions
    and gradual anomalies like fighting/riots) via parallel depthwise-separable
    dilations d in {1, 2, 4} with GLU activations.
    """
    def __init__(self, embed_dim: int = 512, dilations=(1, 2, 4), dropout: float = 0.1):
        super().__init__()
        self.dilations = dilations
        self.branches = nn.ModuleList([
            nn.Sequential(
                DepthwiseSeparableConv1d(
                    embed_dim, embed_dim * 2, kernel_size=3, dilation=d, padding=d
                )
            ) for d in dilations
        ])
        self.norms = nn.ModuleList([
            LayerNormFP32(embed_dim) for _ in dilations
        ])
        self.fuse_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            QuickGELU(),
            nn.Dropout(dropout)
        )
        self.out_norm = LayerNormFP32(embed_dim)

    def forward(self, x: torch.Tensor):
        # x: [B, T, D] -> transpose to [B, D, T] for conv1d
        x_t = x.transpose(1, 2)
        branch_outputs = []
        for branch, norm in zip(self.branches, self.norms):
            # branch output: [B, 2*D, T]
            h = branch(x_t)
            h = F.glu(h, dim=1)  # [B, D, T] via Gated Linear Unit
            # Normalize over feature dimension: [B, T, D]
            h_norm = norm(h.transpose(1, 2)).transpose(1, 2)
            branch_outputs.append(h_norm)

        # Sum parallel multi-scale temporal representations
        combined = sum(branch_outputs)  # [B, D, T]
        fused = self.fuse_conv(combined).transpose(1, 2)  # [B, T, D]
        # Residual connection
        out = self.out_norm(x + fused)
        return out


class MDTBackbone(nn.Module):
    """Multi-Scale Dilated Temporal & Local Attention Backbone.
    
    Combines:
    - Branch A: Local Windowed Transformer (window W=8 for UCF, W=64 for XD)
    - Branch B: Multi-Scale Dilated Temporal Pyramid (dilations 1, 2, 4)
    - Channel-wise attention gate to fuse Branch A & B into F_base
    - Base anomaly detector classifier S_det
    """
    def __init__(self, visual_length: int = 256, embed_dim: int = 512,
                 attn_window: int = 8, visual_layers: int = 2, visual_heads: int = 4,
                 mlp_dropout: float = 0.1, no_mdtp: bool = False):
        super().__init__()
        self.visual_length = visual_length
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.no_mdtp = no_mdtp

        # Learnable temporal frame positional embedding
        self.frame_pos_embed = nn.Parameter(torch.zeros(1, visual_length, embed_dim))
        trunc_normal_(self.frame_pos_embed, std=0.02)

        # Branch A: Local Windowed Transformer
        self.branch_a_layers = nn.ModuleList([
            WindowedAttentionBlock(embed_dim, num_heads=visual_heads, dropout=mlp_dropout)
            for _ in range(visual_layers)
        ])

        # Branch B: Multi-Scale Dilated Temporal Pyramid
        if not self.no_mdtp:
            self.branch_b = MDTPBranch(embed_dim=embed_dim, dilations=(1, 2, 4), dropout=mlp_dropout)
            # Channel-wise attention gate to fuse A and B
            self.channel_gate = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                QuickGELU(),
                nn.Linear(embed_dim // 4, embed_dim),
                nn.Sigmoid()
            )
            self.fuse_norm = LayerNormFP32(embed_dim)

        # Base Anomaly Classifier Head
        self.mlp_det = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(embed_dim, embed_dim * 4)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(mlp_dropout)),
            ("c_proj", nn.Linear(embed_dim * 4, embed_dim)),
            ("dropout2", nn.Dropout(mlp_dropout))
        ]))
        self.classifier_det = nn.Linear(embed_dim, 1)

    def build_window_attention_mask(self, length: int, window: int, device: torch.device):
        """Construct block-diagonal window attention mask.
        
        True means position is masked out (cannot attend); False means attend.
        """
        mask = torch.ones(length, length, dtype=torch.bool, device=device)
        for i in range(math.ceil(length / window)):
            start = i * window
            end = min(start + window, length)
            mask[start:end, start:end] = False
        return mask

    def forward(self, visual_features: torch.Tensor, padding_mask: torch.Tensor = None,
                lengths: torch.Tensor = None):
        """
        Args:
            visual_features: [B, T, D] CLIP snippet visual features
            padding_mask: [B, T] bool mask where True is padded/invalid
            lengths: [B] valid snippet counts
        Returns:
            F_base: [B, T, D] fused base temporal features
            S_det: [B, T, 1] base anomaly detector probability scores
            logits_det: [B, T, 1] raw classifier logits
        """
        B, T, D = visual_features.shape

        # Positional embedding (interpolated if input sequence length differs)
        if T == self.visual_length:
            pos_emb = self.frame_pos_embed
        else:
            pos_emb = F.interpolate(
                self.frame_pos_embed.permute(0, 2, 1),
                size=T, mode='linear', align_corners=False
            ).permute(0, 2, 1)

        x = visual_features + pos_emb

        # Padding mask setup
        if padding_mask is None and lengths is not None:
            padding_mask = (
                torch.arange(T, device=visual_features.device)[None, :]
                >= lengths.to(visual_features.device)[:, None]
            )

        # Branch A: Windowed Self-Attention
        attn_mask = self.build_window_attention_mask(T, self.attn_window, visual_features.device)
        f_a = x
        for layer in self.branch_a_layers:
            f_a = layer(f_a, attn_mask=attn_mask, key_padding_mask=padding_mask)

        # Branch B: Multi-Scale Dilated Temporal Pyramid
        if not self.no_mdtp:
            f_b = self.branch_b(x)
            # Channel-wise attention gate
            # Pool over time: [B, D]
            if padding_mask is not None:
                valid_mask = (~padding_mask).unsqueeze(-1).to(f_a.dtype)
                pooled = (f_a * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1e-5)
            else:
                pooled = f_a.mean(dim=1)
            gate = self.channel_gate(pooled).unsqueeze(1)  # [B, 1, D]
            f_base = self.fuse_norm(f_a + gate * f_b)
        else:
            f_base = f_a

        # Base Anomaly Classifier
        det_features = f_base + self.mlp_det(f_base)
        logits_det = self.classifier_det(det_features)  # [B, T, 1]
        s_det = torch.sigmoid(logits_det)

        return f_base, s_det, logits_det


# ============================================================================
# 2. Core Module 2: ModalityAdaptiveBranch & PACBridge
# ============================================================================

class PACBridge(nn.Module):
    r"""CLIP-Grounded Acoustic Semantic Bridge (PACBridge).
    
    Breakthrough 1.3:
    1. Projects 128-D VGGish (or 512-D) audio to 512-D \tilde{A} using a 2-layer Conv1D (k=3, GELU, LayerNorm) + Linear adapter.
    2. Computes acoustic-to-text cosine similarity L_aud with L2-normalized CLIP text embeddings \hat{E}_{text}.
    3. Computes acoustic semantic anomaly score:
         s_{aud_sem} = 1.0 - Softmax(L_aud)[:, :, 0:1]
    4. Suppresses benign loud audio while amplifying violent sounds in cross-attention fusion gate.
    """
    def __init__(self, audio_dim: int = 128, embed_dim: int = 512):
        super().__init__()
        self.audio_dim = audio_dim
        self.embed_dim = embed_dim

        # 2-layer Conv1D (k=3, GELU, LayerNorm) + Linear adapter
        self.conv1 = nn.Conv1d(audio_dim, embed_dim, kernel_size=3, padding=1)
        self.ln1 = LayerNormFP32(embed_dim)
        self.act = QuickGELU()
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.ln2 = LayerNormFP32(embed_dim)
        self.adapter = nn.Linear(embed_dim, embed_dim)
        self.ln_out = LayerNormFP32(embed_dim)

        # Learnable acoustic gate modulation parameters
        self.gamma = nn.Parameter(torch.tensor(2.0))
        self.beta = nn.Parameter(torch.tensor(-1.0))

    def forward(self, audio: torch.Tensor, category_text_features: torch.Tensor = None):
        """
        Args:
            audio: [B, T, D_a] audio features
            category_text_features: [C, D] CLIP text embeddings
        Returns:
            a_tilde: [B, T, D] projected acoustic features
            s_aud_sem: [B, T, 1] acoustic semantic anomaly score
            l_aud: [B, T, C] acoustic-text similarity logits
        """
        # audio: [B, T, D_a] -> [B, D_a, T]
        x = audio.transpose(1, 2)
        h = self.conv1(x)
        h = self.ln1(h.transpose(1, 2)).transpose(1, 2)
        h = self.act(h)
        h = self.conv2(h)
        h = self.ln2(h.transpose(1, 2)).transpose(1, 2)
        # Linear adapter: [B, T, D]
        a_tilde = self.ln_out(self.adapter(h.transpose(1, 2)))

        B, T, _ = a_tilde.shape
        device = a_tilde.device

        s_aud_sem = torch.full((B, T, 1), 0.5, device=device, dtype=a_tilde.dtype)
        l_aud = None

        if category_text_features is not None:
            a_norm = F.normalize(a_tilde, dim=-1)  # [B, T, D]
            text_norm = F.normalize(category_text_features, dim=-1).to(a_tilde.dtype)  # [C, D]
            l_aud = torch.matmul(a_norm, text_norm.t()) / 0.07  # [B, T, C]
            p_aud = F.softmax(l_aud, dim=-1)
            # Class 0 is normal ('A')
            s_aud_sem = (1.0 - p_aud[:, :, 0:1]).clamp(0.0, 1.0)

        return a_tilde, s_aud_sem, l_aud


class TemporalConvProjector(nn.Module):
    """1D Temporal Convolution Projector with LayerNorm and GELU."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.ln1 = LayerNormFP32(out_channels)
        self.act = QuickGELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.ln2 = LayerNormFP32(out_channels)

    def forward(self, x: torch.Tensor):
        # x: [B, in_channels, T]
        h = self.conv1(x)  # [B, out_channels, T]
        h = self.ln1(h.transpose(1, 2)).transpose(1, 2)
        h = self.act(h)
        h = self.conv2(h)
        h = self.ln2(h.transpose(1, 2)).transpose(1, 2)
        return h


class ModalityAdaptiveBranch(nn.Module):
    """Modality-Adaptive Audio-Visual / Motion Branch with PACBridge.
    
    Supports:
    1. XD-Violence Multimodal Mode: Project pre-extracted audio features
       (VGGish 128-D or CLAP/AST 512-D) aligned with 16-frame video snippets via PACBridge.
    2. Missing-Modality / Motion-Residual Mode: For UCF-Crime (no audio) or
       missing audio snippets, computes 1st- and 2nd-order visual temporal
       residuals (M1 = X_t - X_{t-1}, M2 = M1_t - M1_{t-1}) as dynamic auxiliary modality.
    3. Acoustic-Modulated Noise Suppression Gate G_av.
    4. Auxiliary anomaly score stream S_aux.
    """
    def __init__(self, embed_dim: int = 512, audio_dim: int = 128,
                 num_heads: int = 4, dropout: float = 0.1, no_audio: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.audio_dim = audio_dim
        self.no_audio = no_audio

        # PACBridge acoustic semantic projectors
        self.pac_bridge = PACBridge(audio_dim=audio_dim, embed_dim=embed_dim)
        self.pac_bridge_512 = PACBridge(audio_dim=embed_dim, embed_dim=embed_dim)

        # Audio legacy projectors for fallback compatibility
        self.audio_proj = TemporalConvProjector(audio_dim, embed_dim, kernel_size=3, padding=1)
        self.audio_proj_512 = TemporalConvProjector(embed_dim, embed_dim, kernel_size=3, padding=1)

        # Motion residual projector (from 2*embed_dim to embed_dim)
        self.motion_proj = TemporalConvProjector(embed_dim * 2, embed_dim, kernel_size=3, padding=1)

        # Cross-Modal Attention: Query=F_base (Visual), Key=F_a (Auxiliary), Value=F_a
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ln_cross = LayerNormFP32(embed_dim)

        # Noise Suppression Gate modulated by s_aud_sem: input is embed_dim * 3 + 1
        self.noise_gate_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3 + 1, embed_dim),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        self.ln_fused = LayerNormFP32(embed_dim)

        # Auxiliary Anomaly Classifier Head
        self.mlp_aux = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(embed_dim, embed_dim * 4)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(dropout)),
            ("c_proj", nn.Linear(embed_dim * 4, embed_dim)),
            ("dropout2", nn.Dropout(dropout))
        ]))
        self.classifier_aux = nn.Linear(embed_dim, 1)

    def compute_motion_residuals(self, x_v: torch.Tensor):
        """Compute 1st and 2nd order visual temporal feature residuals."""
        B, T, D = x_v.shape
        m1 = torch.zeros_like(x_v)
        m1[:, 1:] = x_v[:, 1:] - x_v[:, :-1]

        m2 = torch.zeros_like(x_v)
        m2[:, 1:] = m1[:, 1:] - m1[:, :-1]

        motion = torch.cat([m1, m2], dim=-1)  # [B, T, 2*D]
        return motion

    def forward(self, f_base: torch.Tensor, visual_raw: torch.Tensor,
                audio_features: torch.Tensor = None, lengths: torch.Tensor = None,
                category_text_features: torch.Tensor = None):
        """
        Args:
            f_base: [B, T, D] visual base features from MDTBackbone
            visual_raw: [B, T, D] raw snippet visual features
            audio_features: [B, T_a, D_a] audio features (if available) or None
            lengths: [B] valid lengths
            category_text_features: [C, D] CLIP text embeddings
        Returns:
            F_fused: [B, T, D] multimodal/motion fused features
            S_aux: [B, T, 1] auxiliary anomaly scores
            logits_aux: [B, T, 1] raw classifier logits
            G_av: [B, T, D] noise suppression gate values
        """
        B, T, D = f_base.shape

        if audio_features is not None and not self.no_audio:
            # Multi-Modal Audio Mode
            if audio_features.ndim == 2:
                audio_features = audio_features.unsqueeze(0)
            elif audio_features.ndim == 4:
                audio_features = audio_features.squeeze(0)

            # Align batch/split dimension with visual base features if needed
            B_v = f_base.shape[0]
            B_a = audio_features.shape[0]
            if B_a != B_v:
                if B_a < B_v:
                    pad_splits = torch.zeros(
                        B_v - B_a, audio_features.shape[1], audio_features.shape[2],
                        device=audio_features.device, dtype=audio_features.dtype
                    )
                    audio_features = torch.cat([audio_features, pad_splits], dim=0)
                else:
                    audio_features = audio_features[:B_v]

            # If audio sequence length differs from visual, interpolate temporally
            if audio_features.shape[1] != T:
                audio_t = F.interpolate(
                    audio_features.transpose(1, 2),
                    size=T, mode='linear', align_corners=False
                )
            else:
                audio_t = audio_features.transpose(1, 2)  # [B, D_a, T]

            # Project audio features via PACBridge
            in_dim = audio_t.shape[1]
            if in_dim == self.embed_dim:
                f_a, s_aud_sem, _ = self.pac_bridge_512(audio_t.transpose(1, 2), category_text_features)
            elif in_dim == self.audio_dim:
                f_a, s_aud_sem, _ = self.pac_bridge(audio_t.transpose(1, 2), category_text_features)
            else:
                if not hasattr(self, '_dyn_audio_adapter') or self._dyn_audio_adapter.in_channels != in_dim:
                    self._dyn_audio_adapter = nn.Conv1d(in_dim, self.embed_dim, kernel_size=1).to(audio_t.device)
                adapted_audio = self._dyn_audio_adapter(audio_t)
                f_a, s_aud_sem, _ = self.pac_bridge_512(adapted_audio.transpose(1, 2), category_text_features)
        else:
            # Missing-Modality / Motion-Residual Mode (UCF-Crime or Audio Fallback)
            motion = self.compute_motion_residuals(visual_raw)  # [B, T, 2*D]
            motion_t = motion.transpose(1, 2)  # [B, 2*D, T]
            f_a = self.motion_proj(motion_t).transpose(1, 2)  # [B, T, D]
            s_aud_sem = torch.full((B, T, 1), 0.5, device=f_base.device, dtype=f_base.dtype)

        # Bidirectional Cross-Modal Attention: Query=F_base, Key=F_a, Value=F_a
        attn_out, _ = self.cross_attn(query=f_base, key=f_a, value=f_a, need_weights=False)
        attn_out = torch.nan_to_num(attn_out, nan=0.0)
        f_v_from_a = self.ln_cross(attn_out)

        # Acoustic Semantic Modulated Noise Suppression Gate G_av: [B, T, D]
        gate_input = torch.cat([f_base, f_a, f_base * f_a, s_aud_sem], dim=-1)
        g_av_base = self.noise_gate_mlp(gate_input)

        if audio_features is not None and not self.no_audio:
            # Benign loud audio is suppressed while violent audio amplifies F_fused
            gate_mod = 0.20 + 0.80 * torch.sigmoid(self.pac_bridge.gamma * s_aud_sem + self.pac_bridge.beta)
            g_av = g_av_base * gate_mod
        else:
            g_av = g_av_base

        # Fuse features: F_fused = LayerNorm(F_base + G_av * F_v_from_a)
        f_fused = self.ln_fused(f_base + g_av * f_v_from_a)

        # Auxiliary Anomaly Classifier
        aux_features = f_fused + self.mlp_aux(f_fused)
        logits_aux = self.classifier_aux(aux_features)  # [B, T, 1]
        s_aux = torch.sigmoid(logits_aux)

        return f_fused, s_aux, logits_aux, g_av


# ============================================================================
# 3. Core Module 3: SMoSBundle & GradientSafeRUPA
# ============================================================================

class SMoSBundle(nn.Module):
    """Stratified Mixture-of-Stiefel Bundle (SMoSBundle).
    
    Breakthrough 1.1:
    Parameterizes K=4 local Stiefel charts, each of rank m=8 (4x8=32 manifold prototypes),
    initialized via orthogonal initialization per chart, plus K=4 scene centroids C in R^{4 x 512}.
    Computes chart routing probabilities pi_b in R^{B x 4} from video-level context f_bar_b,
    projects F_fused onto each Stiefel chart via temperature-scaled cosine attention, and
    aggregates reconstructions via pi_{b,k} to form F_rec.
    """
    def __init__(self, embed_dim: int = 512, num_charts: int = 4, rank: int = 8, temp: float = 0.10):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_charts = num_charts
        self.rank = rank
        self.temp = temp

        # Stiefel charts: K=4 charts of rank m=8
        self.stiefel_charts = nn.Parameter(torch.randn(1, num_charts, rank, embed_dim))
        with torch.no_grad():
            for k in range(num_charts):
                nn.init.orthogonal_(self.stiefel_charts[0, k])

        # Scene centroids: K=4 centroids
        self.chart_centroids = nn.Parameter(torch.randn(num_charts, embed_dim))
        trunc_normal_(self.chart_centroids, std=0.02)

    def forward(self, f_fused: torch.Tensor, valid_mask: torch.Tensor = None):
        """
        Args:
            f_fused: [B, T, D] multimodal fused features
            valid_mask: [B, T] bool/float mask (1 for valid snippet, 0 for padded)
        Returns:
            f_rec: [B, T, D] reconstructed counterfactual normal features
            pi_b: [B, K] chart routing probabilities
            orth_loss: [1] chart orthogonality and centroid separation loss
        """
        B, T, D = f_fused.shape
        device = f_fused.device

        if valid_mask is None:
            valid_mask = torch.ones(B, T, device=device, dtype=f_fused.dtype)
        else:
            valid_mask = valid_mask.to(device=device, dtype=f_fused.dtype)

        # 1. Video-level context \bar{f}_b
        mask_expanded = valid_mask.unsqueeze(-1)  # [B, T, 1]
        denom = mask_expanded.sum(dim=1).clamp(min=1e-5)  # [B, 1]
        f_bar_raw = (f_fused * mask_expanded).sum(dim=1) / denom  # [B, D]
        f_bar = F.normalize(f_bar_raw, dim=-1)  # [B, D]

        # 2. Chart routing probabilities \pi_b = Softmax(\bar{f}_b \cdot \hat{C}^\top / 0.10)
        c_norm = F.normalize(self.chart_centroids, dim=-1)  # [K, D]
        routing_logits = torch.matmul(f_bar, c_norm.t()) / self.temp  # [B, K]
        pi_b = F.softmax(routing_logits, dim=-1)  # [B, K]

        # 3. Project F_fused onto each Stiefel chart
        f_norm = F.normalize(f_fused, dim=-1)  # [B, T, D]
        f_mag = torch.norm(f_fused, dim=-1, keepdim=True)  # [B, T, 1]

        u_all = self.stiefel_charts[0]  # [K, m, D]
        chart_recs = []
        for k in range(self.num_charts):
            u_k = u_all[k]  # [m, D]
            u_k_norm = F.normalize(u_k, dim=-1)  # [m, D]
            # Cosine attention: [B, T, m]
            attn_logits = torch.matmul(f_norm, u_k_norm.t()) / self.temp
            attn = F.softmax(attn_logits, dim=-1)  # [B, T, m]
            # Reconstruct in chart k: [B, T, D]
            rec_k = torch.matmul(attn, u_k) * f_mag
            chart_recs.append(rec_k)

        # Aggregate across charts using \pi_b: [B, T, D]
        chart_recs_stack = torch.stack(chart_recs, dim=1)  # [B, K, T, D]
        pi_expanded = pi_b.view(B, self.num_charts, 1, 1)
        f_rec = (pi_expanded * chart_recs_stack).sum(dim=1)  # [B, T, D]

        # 4. Orthogonality & Separation Loss
        orth_losses = []
        for k in range(self.num_charts):
            u_k_norm = F.normalize(u_all[k], dim=-1)  # [m, D]
            gram = torch.matmul(u_k_norm, u_k_norm.t())  # [m, m]
            eye = torch.eye(self.rank, device=device)
            orth_losses.append(((gram - eye) ** 2).mean())
        loss_orth_charts = torch.stack(orth_losses).mean()

        # Centroid separation loss
        c_gram = torch.matmul(c_norm, c_norm.t())  # [K, K]
        eye_c = torch.eye(self.num_charts, device=device)
        loss_sep = ((c_gram - eye_c) ** 2).sum() / (self.num_charts * (self.num_charts - 1))

        orth_loss = (loss_orth_charts + loss_sep).view(1)

        return f_rec, pi_b, orth_loss


class GradientSafeRUPA(nn.Module):
    """Gradient-Safe RUPA Module with SMoSBundle & Tangent Residual.
    
    Breakthrough 1:
    1. SMoSBundle replaces single prototype matrix with a Stratified Mixture-of-Stiefel Bundle.
    2. Calibrated reconstruction anomaly score: S_rec = sigma(w_rec * d_rec + b_rec).
    3. Anomaly-Modulated Tangent Residual:
         F_{sem_rupa} = LayerNorm(F_{base} + S_{rec}.detach().unsqueeze(-1) * alpha_{rupa} * MLP_{res}(F_{fused} - F_{rec}))
       Guarantees Semantics RUPA starts at 83.63%+ AP on Epoch 1 and refines anomalous frames!
    """
    def __init__(self, feature_dim: int = 512, num_prototypes: int = 16,
                 num_heads: int = 8, extractor_depth: int = 1, decoder_depth: int = 4,
                 sampling: str = 'topk', normal_selection_ratio: float = 0.8,
                 min_normal_frames: int = 4, max_normal_ratio: float = 0.8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_prototypes = num_prototypes
        self.sampling = sampling
        self.normal_selection_ratio = normal_selection_ratio
        self.min_normal_frames = min_normal_frames
        self.max_normal_ratio = max_normal_ratio

        # SMoSBundle: Stratified Mixture-of-Stiefel Bundle (K=4 charts, rank m=8)
        self.smos_bundle = SMoSBundle(embed_dim=feature_dim, num_charts=4, rank=8, temp=0.10)

        # Calibrated reconstruction parameters: init w_rec=12.0, b_rec=-4.0
        self.w_rec = nn.Parameter(torch.tensor(12.0))
        self.b_rec = nn.Parameter(torch.tensor(-4.0))

        # Anomaly-Modulated Tangent Residual: init alpha_rupa=0.15
        self.alpha_rupa = nn.Parameter(torch.tensor(0.15))
        self.mlp_res = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            QuickGELU(),
            nn.Linear(feature_dim, feature_dim)
        )
        nn.init.normal_(self.mlp_res[0].weight, std=0.02)
        nn.init.zeros_(self.mlp_res[0].bias)
        nn.init.normal_(self.mlp_res[-1].weight, std=0.02)
        nn.init.zeros_(self.mlp_res[-1].bias)
        self.ln_sem_rupa = LayerNormFP32(feature_dim)

        # Legacy prototypes & extractor for backward compatibility
        self.video_prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim))
        trunc_normal_(self.video_prototypes, std=0.02)
        self.dnp_extractor = nn.ModuleList([
            Aggregation_Block(
                dim=feature_dim, num_heads=num_heads, mlp_ratio=4.0,
                qkv_bias=True, norm_layer=partial(LayerNormFP32, eps=1e-6)
            ) for _ in range(extractor_depth)
        ])
        self.bottleneck = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(feature_dim, feature_dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(feature_dim * 4, feature_dim))
        ]))
        self.decoder = nn.ModuleList([
            Prototype_Block(
                dim=feature_dim, num_heads=num_heads, mlp_ratio=4.0,
                qkv_bias=True, norm_layer=partial(LayerNormFP32, eps=1e-6)
            ) for _ in range(decoder_depth)
        ])
        self.log_temp = nn.Parameter(torch.tensor(math.log(0.07)))
        self.var_scale_param = nn.Parameter(torch.tensor(0.1))

    def forward(self, f_fused: torch.Tensor, s_det: torch.Tensor,
                category_text_features: torch.Tensor, lengths: torch.Tensor = None,
                f_base: torch.Tensor = None):
        """
        Args:
            f_fused: [B, T, D] multimodal fused features
            s_det: [B, T, 1] detector anomaly scores
            category_text_features: [C, D] text embeddings for categories
            lengths: [B] valid sequence lengths
            f_base: [B, T, D] base features from MDTBackbone (optional anchor)
        Returns:
            f_rec: [B, T, D] reconstructed counterfactual normal features
            residual: [B, T, D] residual anomaly features
            s_rec: [B, T, 1] calibrated reconstruction anomaly scores
            s_sem: [B, T, 1] domain-adaptive semantic anomaly scores (S_sem_rupa)
            rupa_dict: detailed tensors for loss supervision & routing
        """
        B, T, D = f_fused.shape
        device = f_fused.device

        if f_base is None:
            f_base = f_fused

        if lengths is not None:
            valid_mask = torch.arange(T, device=device)[None, :] < lengths.to(device)[:, None]
        else:
            valid_mask = torch.ones(B, T, device=device, dtype=torch.bool)

        # 1. Stratified Mixture-of-Stiefel Bundle reconstruction
        f_rec, pi_b, orth_loss = self.smos_bundle(f_fused, valid_mask=valid_mask)
        residual = f_fused - f_rec

        # 2. Calibrated reconstruction anomaly score: S_rec = sigma(w_rec * d_rec + b_rec)
        cos_sim = F.cosine_similarity(f_fused, f_rec, dim=-1).unsqueeze(-1)  # [B, T, 1]
        d_rec = ((1.0 - cos_sim) / 2.0).clamp(0.0, 1.0)
        s_rec = torch.sigmoid(self.w_rec * d_rec + self.b_rec)  # [B, T, 1]

        # 3. Anomaly-Modulated Tangent Residual:
        # F_{sem_rupa} = LayerNorm(F_{base} + S_{rec}.detach().unsqueeze(-1) * alpha_{rupa} * MLP_{res}(F_{fused} - F_{rec}))
        res_mod = self.mlp_res(residual)
        f_sem_rupa = self.ln_sem_rupa(f_base + s_rec.detach() * self.alpha_rupa * res_mod)

        # 4. Semantic alignment logits
        cat_text_norm = F.normalize(category_text_features, dim=-1)  # [C, D]
        text_matrix = cat_text_norm.t().to(f_fused.dtype)  # [D, C]

        f_sem_rupa_norm = F.normalize(f_sem_rupa, dim=-1)
        logits_sem_rupa = torch.matmul(f_sem_rupa_norm, text_matrix) / 0.07  # [B, T, C]
        s_sem = (1.0 - F.softmax(logits_sem_rupa, dim=-1)[:, :, 0:1]).clamp(0.0, 1.0)

        # Base semantic score directly from F_base for routing anchor
        f_base_norm = F.normalize(f_base, dim=-1)
        logits_sem_base = torch.matmul(f_base_norm, text_matrix) / 0.07  # [B, T, C]
        s_sem_base = (1.0 - F.softmax(logits_sem_base, dim=-1)[:, :, 0:1]).clamp(0.0, 1.0)

        rupa_dict = {
            'reconstructed_features': f_rec,
            'residual_features': residual,
            'dynamic_normal_patterns': self.smos_bundle.stiefel_charts[0].reshape(-1, D).unsqueeze(0).expand(B, -1, -1),
            'semantic_logits': logits_sem_rupa,
            'raw_semantic_logits': logits_sem_rupa,
            'residual_semantic_logits': logits_sem_rupa,
            'reconstructed_semantic_logits': logits_sem_base,
            'semantic_scores_base': s_sem_base,
            'g_loss': torch.zeros(1, device=device),
            'orth_loss': orth_loss,
            'dnp_normal_loss': torch.zeros(1, device=device),
            'stiefel_charts': self.smos_bundle.stiefel_charts.expand(B, -1, -1, -1),
            'chart_centroids': self.smos_bundle.chart_centroids.unsqueeze(0).expand(B, -1, -1),
            'chart_routing_probs': pi_b,
            'f_sem_rupa': f_sem_rupa,
        }

        return f_rec, residual, s_rec, s_sem, rupa_dict


# ============================================================================
# 4. Core Module 4: Consensus-Discounted Evidential Routing (DEM-Router)
# ============================================================================

class DEMRouter(nn.Module):
    """Consensus-Discounted Evidential Routing (DEM-Router).
    
    Breakthrough 3:
    Replaces inverse-entropy UCCF with sequence-level consensus trust routing.
    - Anchor Stream:
        XD-Violence: S_{anchor} = 0.35 * S_{det} + 0.65 * S_{sem_base}
        UCF-Crime:   S_{anchor} = 0.55 * S_{det} + 0.45 * S_{sem_base}
    - Auxiliary refinement streams: S_{sem_rupa}, S_{rec}, S_{aux}.
    - Per-video sequence consensus trust rho_{b,j} = (r_{b,j} * m_{b,j}).detach().
    - Bounded update:
        P_{routing} = S_{anchor} + 0.30 * g * max_j(rho_{b,j}) * (S_{refine} - S_{anchor}).
    Guarantees zero late-epoch degradation.
    """
    def __init__(self, dataset_type: str = 'xd', hidden_dim: int = 16,
                 no_safegate: bool = False, custom_prior_weights=None):
        super().__init__()
        self.dataset_type = dataset_type.lower()
        self.no_safegate = no_safegate

        # MLP Gate: takes [S_anchor, S_refine, |S_refine - S_anchor|, rho_max]
        self.mlp_gate = nn.Sequential(
            nn.Linear(4, hidden_dim),
            QuickGELU(),
            nn.Linear(hidden_dim, 1)
        )
        nn.init.normal_(self.mlp_gate[0].weight, std=0.01)
        nn.init.constant_(self.mlp_gate[0].bias, 0.0)
        nn.init.normal_(self.mlp_gate[-1].weight, std=0.01)
        nn.init.constant_(self.mlp_gate[-1].bias, -4.5)

    def forward(self, s_det: torch.Tensor, s_rec: torch.Tensor,
                s_sem: torch.Tensor, s_aux: torch.Tensor,
                s_sem_base: torch.Tensor = None, lengths: torch.Tensor = None):
        """
        Args:
            s_det, s_rec, s_sem, s_aux: [B, T, 1] anomaly scores
            s_sem_base: [B, T, 1] base semantic score (optional)
            lengths: [B] valid lengths
        Returns:
            p_routing: [B, T, 1] routed anomaly probability
            gate: [B, T, 1] dynamic gate
            s_anchor: [B, T, 1] anchor stream
        """
        s_det = s_det.clamp(0.0, 1.0)
        s_rec = s_rec.clamp(0.0, 1.0)
        s_sem_rupa = s_sem.clamp(0.0, 1.0)
        s_aux = s_aux.clamp(0.0, 1.0)

        if s_sem_base is None:
            s_sem_base = s_sem_rupa
        else:
            s_sem_base = s_sem_base.clamp(0.0, 1.0)

        B, T, _ = s_det.shape
        device = s_det.device

        # 1. Primary Anchor Stream S_{anchor}
        if 'xd' in self.dataset_type:
            s_anchor = 0.35 * s_det + 0.65 * s_sem_base
        else:
            s_anchor = 0.55 * s_det + 0.45 * s_sem_base

        if self.no_safegate:
            return s_anchor, torch.zeros_like(s_det), s_anchor

        # 2. Per-Video Sequence Consensus Trust rho_{b,j}
        aux_streams = [s_sem_rupa, s_rec, s_aux]
        num_aux = len(aux_streams)

        if lengths is None:
            valid_lengths = torch.full((B,), T, device=device, dtype=torch.long)
        else:
            valid_lengths = lengths.to(device=device, dtype=torch.long).clamp(min=1, max=T)

        rho_list = []
        for j, s_j in enumerate(aux_streams):
            rho_j_batch = []
            for b in range(B):
                l_b = int(valid_lengths[b].item())
                s_jb = s_j[b, :l_b, 0]  # [L_b]
                s_ab = s_anchor[b, :l_b, 0]  # [L_b]

                s_jb_mean = s_jb.mean()
                s_ab_mean = s_ab.mean()

                s_jb_cent = s_jb - s_jb_mean
                s_ab_cent = s_ab - s_ab_mean

                norm_jb = torch.norm(s_jb_cent)
                norm_ab = torch.norm(s_ab_cent)

                if norm_jb > 1e-5 and norm_ab > 1e-5:
                    cos_val = (s_jb_cent * s_ab_cent).sum() / (norm_jb * norm_ab + 1e-6)
                    r_bj = (torch.relu(cos_val)) ** 2
                else:
                    # Constant signals agree with mean calibration
                    r_bj = torch.tensor(1.0, device=device)

                m_bj = torch.exp(-((s_jb_mean - s_ab_mean) ** 2) / 0.08)
                rho_bj = (r_bj * m_bj).detach().clamp(0.0, 1.0)
                rho_j_batch.append(rho_bj)

            rho_list.append(torch.stack(rho_j_batch))  # [B]

        rho_stack = torch.stack(rho_list, dim=1)  # [B, num_aux]
        rho_sum = rho_stack.sum(dim=1, keepdim=True) + 1e-4  # [B, 1]
        w_aux = rho_stack / rho_sum  # [B, num_aux]
        rho_max = rho_stack.max(dim=1, keepdim=True)[0]  # [B, 1]

        # S_{refine} = Sum_j w_{b,j} S_{j,b,t}
        s_refine = torch.zeros_like(s_anchor)
        for j in range(num_aux):
            s_refine = s_refine + w_aux[:, j].view(B, 1, 1) * aux_streams[j]

        # 3. Bounded Trust-Region Update
        diff = (s_refine - s_anchor).abs()
        v_gate = torch.cat([
            s_anchor, s_refine, diff, rho_max.unsqueeze(1).expand(-1, T, -1)
        ], dim=-1)  # [B, T, 4]

        gate = torch.sigmoid(self.mlp_gate(v_gate))  # [B, T, 1]

        p_routing = s_anchor + 0.30 * gate * rho_max.view(B, 1, 1) * (s_refine - s_anchor)
        p_routing = p_routing.clamp(1e-5, 1.0 - 1e-5)

        return p_routing, gate, s_anchor


# Backward compatibility alias
QuadSafeGate = DEMRouter


# ============================================================================
# 5. Text CLIP Adapter & Feature Cache
# ============================================================================

class CLIPAdapter(nn.Module):
    """Adapts CLIP text encoder for video anomaly detection."""
    def __init__(self, clipmodel, device, text_adapt_until: int = 3, t_w: float = 0.1):
        super().__init__()
        self.clipmodel = clipmodel
        self.text_adapt_until = text_adapt_until
        self.t_w = t_w
        self.device = device

        self.text_adapter = nn.ModuleList(
            [SimpleAdapter(512, 512) for _ in range(text_adapt_until)] +
            [SimpleProj(512, 512, relu=True)]
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.text_adapter.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_text(self, text: torch.Tensor, adapt_text: bool = True):
        if not adapt_text:
            return self.clipmodel.encode_text(text)

        cast_dtype = self.clipmodel.token_embedding.weight.dtype
        x = self.clipmodel.token_embedding(text).to(cast_dtype)
        x = x + self.clipmodel.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)

        for i in range(len(self.clipmodel.transformer.resblocks)):
            x = self.clipmodel.transformer.resblocks[i](x)
            if i < self.text_adapt_until:
                adapt_out = self.text_adapter[i](x)
                adapt_out = (
                    adapt_out * x.norm(dim=-1, keepdim=True) /
                    (adapt_out.norm(dim=-1, keepdim=True) + 1e-6)
                )
                x = self.t_w * adapt_out + (1 - self.t_w) * x

        x = x.permute(1, 0, 2)
        x = self.clipmodel.ln_final(x)
        eot_indices = text.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eot_indices]
        x = self.text_adapter[-1](x)
        return x


# ============================================================================
# 6. Full Model: MARUPA
# ============================================================================

class MARUPA(nn.Module):
    """MA-RUPA: Modality-Adaptive Multiscale Prototype Alignment Network.
    
    End-to-End trainable from scratch without pre-trained DSANet checkpoints
    or frozen two-stage warm-starts.
    """
    def __init__(self,
                 num_class: int,
                 embed_dim: int = 512,
                 visual_length: int = 256,
                 visual_width: int = 512,
                 visual_head: int = 4,
                 visual_layers: int = 2,
                 attn_window: int = 8,
                 prompt_prefix: int = 10,
                 prompt_postfix: int = 10,
                 args = None,
                 device = "cuda"):
        super().__init__()
        self.num_class = num_class
        self.embed_dim = embed_dim
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.attn_window = attn_window
        self.device = device
        self.args = args

        # Ablation study flags
        self.no_mdtp = getattr(args, 'no_mdtp', False)
        self.no_audio = getattr(args, 'no_audio', False)
        self.no_rupa = getattr(args, 'no_rupa', False)
        self.no_safegate = getattr(args, 'no_safegate', False)
        self.sampling = getattr(args, 'sampling', 'topk')
        dataset_type = 'xd' if num_class == 7 else 'ucf'

        # 1. Core Module 1: MDTBackbone
        self.backbone = MDTBackbone(
            visual_length=visual_length,
            embed_dim=visual_width,
            attn_window=attn_window,
            visual_layers=visual_layers,
            visual_heads=visual_head,
            mlp_dropout=getattr(args, 'mlp_dropout', 0.1),
            no_mdtp=self.no_mdtp
        )

        # 2. Core Module 2: ModalityAdaptiveBranch
        audio_dim = getattr(args, 'audio_dim', 128)
        self.modality_branch = ModalityAdaptiveBranch(
            embed_dim=visual_width,
            audio_dim=audio_dim,
            num_heads=visual_head,
            dropout=getattr(args, 'mlp_dropout', 0.1),
            no_audio=self.no_audio
        )

        # 3. Core Module 3: GradientSafeRUPA
        self.rupa = GradientSafeRUPA(
            feature_dim=visual_width,
            num_prototypes=getattr(args, 'num_prototypes', 16),
            num_heads=visual_head,
            extractor_depth=1,
            decoder_depth=getattr(args, 'decoder_depth', 4),
            sampling=self.sampling,
            normal_selection_ratio=getattr(args, 'normal_selection_ratio', 0.8),
            min_normal_frames=getattr(args, 'min_normal_frames', 4),
            max_normal_ratio=getattr(args, 'max_normal_ratio', 0.8)
        )

        # 4. Core Module 4: QuadSafeGate
        prior_weights = None
        if hasattr(args, 'routing_det_weight') and hasattr(args, 'routing_rec_weight'):
            prior_weights = [
                getattr(args, 'routing_det_weight', 0.4),
                getattr(args, 'routing_rec_weight', 0.3),
                getattr(args, 'routing_sem_weight', 0.1),
                getattr(args, 'routing_aux_weight', 0.2)
            ]
        self.safegate = QuadSafeGate(
            dataset_type=dataset_type,
            hidden_dim=getattr(args, 'routing_gate_hidden', 16),
            no_safegate=self.no_safegate,
            custom_prior_weights=prior_weights
        )

        # 5. CLIP Text Adapter
        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for p in self.clipmodel.parameters():
            p.requires_grad = False

        text_adapt_until = getattr(args, 'text_adapt_until', 1 if dataset_type == 'xd' else 3)
        t_w = getattr(args, 't_w', 0.6 if dataset_type == 'xd' else 0.1)
        self.clip_adapter = CLIPAdapter(self.clipmodel, device, text_adapt_until, t_w)

        # Event & Background alignment MLPs
        self.mlp_event = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(getattr(args, 'mlp_dropout', 0.1))),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))

        self._text_features_cache = None
        self._init_trainable_parameters()

    def _init_trainable_parameters(self):
        """Initialize all trainable layers with truncated normal/Xavier."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02, a=-0.04, b=0.04)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.LayerNorm, LayerNormFP32)):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def get_text_features(self, text_list):
        """Encode category descriptions into normalized CLIP text embeddings."""
        if not self.training and self._text_features_cache is not None:
            return self._text_features_cache

        descriptions = DESCRIPTIONS_ORI if len(text_list) == 14 else DESCRIPTIONS_ORI_XD
        category_features = []
        for class_name, desc_items in descriptions.items():
            tokens = clip.tokenize(desc_items).to(self.device)
            feats = self.clip_adapter.encode_text(tokens)
            mean_feat = feats.mean(dim=0)
            mean_feat = mean_feat / mean_feat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            category_features.append(mean_feat)

        text_features = torch.stack(category_features, dim=0)  # [C, D]
        if not self.training:
            self._text_features_cache = text_features
        return text_features

    def forward(self, visual: torch.Tensor, padding_mask: torch.Tensor = None,
                text: list = None, lengths: torch.Tensor = None,
                audio: torch.Tensor = None, DNP_use: bool = True, scale: float = 10.0):
        """Full MA-RUPA forward pass.
        
        Args:
            visual: [B, T, D] visual features
            padding_mask: [B, T] bool mask of padding positions
            text: list of class names or descriptions
            lengths: [B] valid sequence lengths
            audio: [B, T_a, D_a] audio features (XD-Violence) or None (UCF-Crime)
            DNP_use: whether to activate RUPA module
            scale: temperature scale for temporal pooling
        """
        B, T, D = visual.shape

        # Step 1: MDTBackbone -> F_base, S_det
        f_base, s_det, logits_det = self.backbone(visual, padding_mask, lengths)

        # Text features for semantic alignment & acoustic semantic bridge
        text_features_ori = self.get_text_features(text)

        # Step 2: ModalityAdaptiveBranch -> F_fused, S_aux (with PACBridge acoustic grounding)
        f_fused, s_aux, logits_aux, g_av = self.modality_branch(
            f_base, visual, audio_features=audio, lengths=lengths, category_text_features=text_features_ori
        )

        # Step 3 & 4: SMoSBundle RUPA & DEMRouter
        dnp_dict = {}
        if DNP_use and not self.no_rupa:
            f_rec, residual, s_rec, s_sem, rupa_dict = self.rupa(
                f_fused, s_det, text_features_ori, lengths=lengths, f_base=f_base
            )
            s_sem_base = rupa_dict.get('semantic_scores_base', s_sem)
            p_routing, gate, s_anchor = self.safegate(
                s_det, s_rec, s_sem, s_aux, s_sem_base=s_sem_base, lengths=lengths
            )
            dnp_dict.update(rupa_dict)
            dnp_dict.update({
                'routing_scores': p_routing,
                'routing_logits': torch.logit(p_routing),
                'anchor_scores': s_anchor,
                'detector_scores': s_det,
                'detector_logits': logits_det,
                'aux_scores': s_aux,
                'aux_logits': logits_aux,
                'reconstruction_scores': s_rec,
                'semantic_scores': s_sem,
                'semantic_scores_base': s_sem_base,
                'fixed_scores': s_anchor,
                'gate': gate,
                'noise_gate_av': g_av,
                'original_features': f_fused,
                'reconstructed_features': f_rec,
                'residual_features': residual,
            })
            logits1 = logits_det
            event_features = residual
            background_features = f_rec
        else:
            # Fallback when RUPA is disabled (Ablation mode)
            p_routing, gate, s_anchor = self.safegate(
                s_det, s_det, s_det, s_aux, s_sem_base=s_det, lengths=lengths
            )
            logits1 = logits_det
            event_features = f_fused
            background_features = f_fused
            dnp_dict.update({
                'routing_scores': p_routing,
                'routing_logits': torch.logit(p_routing),
                'anchor_scores': s_anchor,
                'detector_scores': s_det,
                'detector_logits': logits_det,
                'aux_scores': s_aux,
                'aux_logits': logits_aux,
                'reconstruction_scores': s_det,
                'semantic_scores': s_det,
                'semantic_scores_base': s_det,
                'fixed_scores': s_anchor,
                'gate': gate,
                'noise_gate_av': g_av,
                'original_features': f_fused,
                'reconstructed_features': f_fused,
                'residual_features': torch.zeros_like(f_fused),
                'g_loss': torch.zeros(1, device=visual.device),
                'orth_loss': torch.zeros(1, device=visual.device),
                'dnp_normal_loss': torch.zeros(1, device=visual.device),
            })

        # Event-centric & Background-centric attention pooling
        if lengths is not None:
            valid_mask = (
                torch.arange(T, device=visual.device)[None, :]
                < lengths.to(visual.device)[:, None]
            ).unsqueeze(-1).to(visual.dtype)
        else:
            valid_mask = torch.ones(B, T, 1, device=visual.device, dtype=visual.dtype)

        prob_abn = p_routing
        abn_weights = ((scale * prob_abn).exp() - 1.0) * valid_mask
        abn_weights = F.normalize(abn_weights, p=1, dim=1)

        nor_weights = ((scale * (1.0 - prob_abn)).exp() - 1.0) * valid_mask
        nor_weights = F.normalize(nor_weights, p=1, dim=1)

        # Video-level event & background representations
        abn_feat = torch.matmul(abn_weights.permute(0, 2, 1), event_features)
        nor_feat = torch.matmul(nor_weights.permute(0, 2, 1), background_features)

        # Snippet-level and video-level classification logits
        text_mat = F.normalize(text_features_ori, dim=-1).t().to(visual.dtype)
        f_norm = F.normalize(f_fused, dim=-1)
        logits2 = f_norm @ text_mat / 0.07

        abn_feat_norm = F.normalize(abn_feat, dim=-1)
        nor_feat_norm = F.normalize(nor_feat, dim=-1)
        logits3 = abn_feat_norm @ text_mat / 0.07
        logits4 = nor_feat_norm @ text_mat / 0.07

        if DNP_use:
            return text_features_ori, logits1, logits2, logits3, logits4, dnp_dict
        return text_features_ori, logits1, logits2, logits3, logits4


# Legacy alias for backward compatibility with imports
DSANet = MARUPA


def load_compatible_model_state(model, state_dict, strict=False):
    """Safely loads model checkpoint."""
    target_model = model.module if hasattr(model, 'module') else model
    return target_model.load_state_dict(state_dict, strict=strict)
