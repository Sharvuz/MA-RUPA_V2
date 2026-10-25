"""MA-RUPA Advanced Losses & Mathematical Formulations.

Implements Breakthrough 2:
1. Perona-Malik Anisotropic Shock-Preserving Loss (L_Shock-TV) + 1D Haar Wavelet Contrast Loss (L_Wavelet).
2. Stochastic Peak-Drop MIL Loss (DynamicTopKMIL with p_drop=0.30).
3. One-Sided Geodesic Halo Expansion Loss (L_Halo with lambda_halo=0.12).
4. Multi-class Semantic Alignment (CLASM), Background Prototype, and Consistency Losses.
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_stochastic_topk_mil_loss(scores: torch.Tensor,
                                     binary_labels: torch.Tensor,
                                     lengths: torch.Tensor,
                                     device: torch.device,
                                     topk_ratio: int = 16,
                                     label_smoothing: float = 0.0,
                                     training: bool = True,
                                     p_drop: float = 0.30,
                                     drop_ratio: float = 0.20) -> torch.Tensor:
    """Stochastic Peak-Drop Multiple Instance Learning (MIL) Binary Cross-Entropy Loss.
    
    Breaks the 0.0142 single-frame shortcut:
    During training, for abnormal bags (y_b = 1) with k_b >= 4, randomly drops 20%
    of the highest-scoring snippets and replaces them with adjacent candidates [k_b : k_b + Delta k]
    with probability p_drop = 0.30.
    """
    instance_scores = []
    binary_labels = binary_labels.to(device)
    scores_flat = scores.squeeze(-1)  # [B, T]
    B = scores.shape[0]

    for i in range(B):
        l_i = int(lengths[i].item())
        k_b = max(1, int(l_i / topk_ratio) + 1)
        valid_scores = scores_flat[i, :l_i]
        y_i = float(binary_labels[i].item())

        # Stochastic Peak-Drop for abnormal bags
        if training and y_i >= 0.5 and k_b >= 4 and (random.random() < p_drop):
            n_drop = max(1, int(drop_ratio * k_b))
            k_total = min(l_i, k_b + n_drop)
            topk_vals, _ = torch.topk(valid_scores, k=k_total, largest=True)
            
            # Drop n_drop items from the top k_b
            perm = torch.randperm(k_b, device=device)
            keep_top_idx = perm[n_drop:]
            surviving_top = topk_vals[keep_top_idx]
            replacement = topk_vals[k_b:k_total]
            selected_snippets = torch.cat([surviving_top, replacement], dim=0)
            instance_scores.append(selected_snippets.mean().view(1))
        else:
            topk_vals, _ = torch.topk(valid_scores, k=min(k_b, l_i), largest=True)
            instance_scores.append(topk_vals.mean().view(1))

    instance_scores = torch.cat(instance_scores, dim=0)
    targets = binary_labels.float()
    if label_smoothing > 0.0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing

    loss = F.binary_cross_entropy(instance_scores.clamp(1e-6, 1.0 - 1e-6), targets)
    return loss


# Backward-compatible alias
compute_topk_mil_loss = compute_stochastic_topk_mil_loss


def compute_shock_tv_loss(scores: torch.Tensor,
                          features: torch.Tensor,
                          lengths: torch.Tensor,
                          delta_f_scale: float = 0.20,
                          delta_s_scale: float = 0.15) -> torch.Tensor:
    """Perona-Malik Anisotropic Shock-Preserving Smoothness Loss (L_Shock-TV).
    
    Conductance:
      g_{b,t} = exp(-||F_{b,t} - F_{b,t-1}||^2 / 0.20) * (1 / (1 + (|S_{b,t} - S_{b,t-1}| / 0.15)^2))
    When a sharp anomaly spike occurs (|Delta S| > 0.15), g_{b,t} -> 0, completely preventing peak clipping!
    """
    B = scores.shape[0]
    scores_flat = scores.squeeze(-1)  # [B, T]
    features_norm = F.normalize(features, dim=-1)  # [B, T, D]
    
    per_bag_losses = []
    for b in range(B):
        l_b = int(lengths[b].item())
        if l_b > 1:
            s_b = scores_flat[b, :l_b]  # [L_b]
            f_b = features_norm[b, :l_b]  # [L_b, D]

            diff_s = s_b[1:] - s_b[:-1]  # [L_b - 1]
            diff_f_sq = ((f_b[1:] - f_b[:-1]) ** 2).sum(dim=-1)  # [L_b - 1]

            # Lorentzian shock conductance
            g_b = torch.exp(-diff_f_sq / delta_f_scale) / (1.0 + (diff_s.abs() / delta_s_scale) ** 2)
            shock_loss = (g_b.detach() * (diff_s ** 2)).mean()
            per_bag_losses.append(shock_loss)

    if len(per_bag_losses) == 0:
        return torch.tensor(0.0, device=scores.device)
    return torch.stack(per_bag_losses).mean()


def compute_haar_wavelet_loss(scores: torch.Tensor,
                              binary_labels: torch.Tensor,
                              lengths: torch.Tensor,
                              hinge_margin: float = 0.25) -> torch.Tensor:
    """1D Haar Wavelet Contrast Loss (L_Wavelet).
    
    - Normal bags (y_b = 0): Penalize mean squared Haar detail 0.5 * (S_{b,t} - S_{b,t-1})^2.
    - Abnormal bags (y_b = 1): Enforce max_{t < L_b} |S_{b,t} - S_{b,t-1}| >= 0.25 via hinge loss.
    """
    B = scores.shape[0]
    scores_flat = scores.squeeze(-1)
    binary_labels = binary_labels.to(scores.device)

    normal_losses = []
    abnormal_losses = []

    for b in range(B):
        l_b = int(lengths[b].item())
        if l_b > 1:
            diff_s = (scores_flat[b, 1:l_b] - scores_flat[b, :l_b - 1]).abs()  # [L_b - 1]
            y_b = float(binary_labels[b].item())

            if y_b < 0.5:
                # Normal bag: suppress fluctuations
                detail = 0.5 * (diff_s ** 2).mean()
                normal_losses.append(detail)
            else:
                # Abnormal bag: enforce prominent temporal transition
                max_jump = diff_s.max()
                hinge = F.relu(hinge_margin - max_jump)
                abnormal_losses.append(hinge)

    total_loss = torch.tensor(0.0, device=scores.device)
    count = 0
    if len(normal_losses) > 0:
        total_loss = total_loss + torch.stack(normal_losses).mean()
        count += 1
    if len(abnormal_losses) > 0:
        total_loss = total_loss + torch.stack(abnormal_losses).mean()
        count += 1

    if count > 0:
        return total_loss / count
    return torch.tensor(0.0, device=scores.device)


def compute_geodesic_halo_loss(scores: torch.Tensor,
                               features: torch.Tensor,
                               binary_labels: torch.Tensor,
                               lengths: torch.Tensor,
                               sigma_t: float = 3.0,
                               halo_threshold: float = 0.25,
                               halo_scale: float = 0.85) -> torch.Tensor:
    """One-Sided Geodesic Halo Expansion Loss (L_Halo).
    
    For each abnormal bag b, let Omega_{core} be the Top-k_core highest scoring snippets.
    Expands anomaly coverage into adjacent semantically-identical onset/offset snippets via:
      q^{halo}_{b,t} = max_{t* in Omega_{core}} [ exp(-(t - t*)^2 / (2 * sigma_t^2)) * max(0, cos(F_t, F_{t*}))^4 ]
    Applies a one-sided upward hinge:
      L_Halo = Mean_{b in B_abn} [ Sum(I(q > 0.25) * q * max(0, 0.85 * s*_b * q - S_{b,t})) / Normalizer ]
    Never pushes peaks down—only lifts adjacent halo snippets!
    """
    B = scores.shape[0]
    scores_flat = scores.squeeze(-1)  # [B, T]
    features_norm = F.normalize(features, dim=-1)  # [B, T, D]
    binary_labels = binary_labels.to(scores.device)

    abnormal_halo_losses = []

    for b in range(B):
        y_b = float(binary_labels[b].item())
        if y_b < 0.5:
            continue

        l_b = int(lengths[b].item())
        if l_b < 2:
            continue

        s_b = scores_flat[b, :l_b]  # [L_b]
        f_b = features_norm[b, :l_b]  # [L_b, D]

        k_core = max(1, l_b // 16)
        topk_scores, topk_indices = torch.topk(s_b, k=min(k_core, l_b), largest=True)
        s_star = topk_scores.mean().detach()

        core_anchors = topk_indices  # [k_core]
        f_core = f_b[core_anchors]  # [k_core, D]

        # Time indices: [L_b, 1] vs [1, k_core]
        t_all = torch.arange(l_b, device=scores.device).unsqueeze(1).float()  # [L_b, 1]
        t_core = core_anchors.unsqueeze(0).float()  # [1, k_core]

        # Temporal distance kernel: [L_b, k_core]
        time_kernel = torch.exp(-((t_all - t_core) ** 2) / (2.0 * (sigma_t ** 2)))

        # Feature similarity kernel: [L_b, k_core]
        cos_sim = torch.matmul(f_b, f_core.t()).clamp(min=0.0)  # [L_b, k_core]
        feat_kernel = cos_sim ** 4

        # Affinity to nearest core anchor
        q_halo, _ = torch.max(time_kernel * feat_kernel, dim=1)  # [L_b]
        q_halo = q_halo.detach()

        # One-sided upward hinge
        halo_mask = q_halo > halo_threshold
        if halo_mask.any():
            target_scores = halo_scale * s_star * q_halo
            hinge = F.relu(target_scores - s_b)
            loss_b = (halo_mask.float() * q_halo * hinge).sum() / (
                (halo_mask.float() * q_halo).sum() + 1e-6
            )
            abnormal_halo_losses.append(loss_b)

    if len(abnormal_halo_losses) == 0:
        return torch.tensor(0.0, device=scores.device)
    return torch.stack(abnormal_halo_losses).mean()


def compute_temporal_smoothness_loss(scores: torch.Tensor,
                                     lengths: torch.Tensor,
                                     lambda_sparse: float = 8e-5) -> torch.Tensor:
    """Standard isotropic temporal total variation and sparsity loss (baseline fallback)."""
    smooth_loss = 0.0
    sparse_loss = 0.0
    B = scores.shape[0]
    scores_flat = scores.squeeze(-1)

    for i in range(B):
        l_i = int(lengths[i].item())
        if l_i > 1:
            diff = scores_flat[i, 1:l_i] - scores_flat[i, :l_i - 1]
            smooth_loss += (diff ** 2).mean()
            sparse_loss += scores_flat[i, :l_i].mean()

    return (smooth_loss / max(1, B)) + lambda_sparse * (sparse_loss / max(1, B))


def compute_residual_event_loss(residual_logits: torch.Tensor,
                                labels: torch.Tensor,
                                lengths: torch.Tensor,
                                device: torch.device) -> torch.Tensor:
    """Aligns anomaly residual features with fine-grained anomaly text prompts."""
    anomaly_mask = labels[:, 1:].sum(dim=1) > 0
    if not torch.any(anomaly_mask):
        return torch.tensor(0.0, device=device)

    sub_logits = residual_logits[anomaly_mask]
    sub_labels = labels[anomaly_mask].to(device)
    sub_lengths = lengths[anomaly_mask]

    labels_norm = sub_labels / sub_labels.sum(dim=1, keepdim=True).clamp_min(1e-6)

    instance_logits = []
    for i in range(sub_logits.shape[0]):
        l_i = int(sub_lengths[i].item())
        k_count = max(1, int(l_i / 16) + 1)
        topk_vals, _ = torch.topk(sub_logits[i, :l_i], k=min(k_count, l_i), largest=True, dim=0)
        instance_logits.append(topk_vals.mean(dim=0, keepdim=True))

    instance_logits = torch.cat(instance_logits, dim=0)
    loss = -torch.mean(torch.sum(labels_norm * F.log_softmax(instance_logits, dim=-1), dim=-1))
    return loss


def compute_clasm_loss(logits: torch.Tensor,
                       labels: torch.Tensor,
                       lengths: torch.Tensor,
                       device: torch.device,
                       k_ratio: int = 16,
                       epsilon: float = 0.05) -> torch.Tensor:
    """Snippet-level multi-class MIL cross-entropy loss against fine-grained categories."""
    labels_norm = labels.to(device) / labels.to(device).sum(dim=1, keepdim=True).clamp_min(1e-6)
    num_classes = logits.shape[-1]
    labels_sm = (1.0 - epsilon) * labels_norm + (epsilon / num_classes)

    instance_logits = []
    for i in range(logits.shape[0]):
        l_i = int(lengths[i].item())
        k_count = max(1, int(l_i / k_ratio) + 1)
        topk_vals, _ = torch.topk(logits[i, :l_i], k=min(k_count, l_i), largest=True, dim=0)
        instance_logits.append(topk_vals.mean(dim=0, keepdim=True))

    instance_logits = torch.cat(instance_logits, dim=0)
    milloss = -torch.mean(torch.sum(labels_sm * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss


def compute_bkg_loss(logits4: torch.Tensor,
                     device: torch.device,
                     num_classes: int = 7,
                     epsilon: float = 0.05) -> torch.Tensor:
    """Background-centric loss forcing reconstructed prototypes to classify as class 0 (Normal)."""
    nor_logits = logits4.squeeze(1)  # [B, C]
    target = torch.full_like(nor_logits, epsilon / num_classes, device=device)
    target[:, 0] = 1.0 - epsilon + (epsilon / num_classes)
    loss = -torch.mean(torch.sum(target * F.log_softmax(nor_logits, dim=1), dim=1))
    return loss


def compute_consistency_loss(p_routing: torch.Tensor,
                             original_features: torch.Tensor,
                             reconstructed_features: torch.Tensor,
                             lengths: torch.Tensor,
                             device: torch.device) -> torch.Tensor:
    """Enforces consistency between detector routing anomaly score and prototype reconstruction error."""
    cos_sim = F.cosine_similarity(original_features, reconstructed_features, dim=-1)  # [B, T]
    recon_error = ((1.0 - cos_sim) / 2.0).clamp(0.0, 1.0)
    pred_score = p_routing.squeeze(-1)  # [B, T]

    B, T = pred_score.shape
    mask = torch.arange(T, device=device)[None, :] < lengths.to(device)[:, None]

    if not torch.any(mask):
        return torch.tensor(0.0, device=device)

    return F.mse_loss(pred_score[mask], recon_error[mask])


def compute_event_loss(logits3: torch.Tensor,
                       labels: torch.Tensor,
                       device: torch.device,
                       epsilon: float = 0.05) -> torch.Tensor:
    """Event-centric loss matching pooled anomaly features against true crime class."""
    anom_mask = labels[:, 1:].sum(dim=1) > 0
    if not torch.any(anom_mask):
        return torch.tensor(0.0, device=device)

    sub_logits = logits3[anom_mask].squeeze(1)  # [B_anom, C]
    sub_labels = labels[anom_mask].to(device)
    num_classes = sub_logits.shape[-1]

    labels_norm = sub_labels / sub_labels.sum(dim=1, keepdim=True).clamp_min(1e-6)
    labels_sm = (1.0 - epsilon) * labels_norm + (epsilon / num_classes)
    loss = -torch.mean(torch.sum(labels_sm * F.log_softmax(sub_logits, dim=1), dim=1))
    return loss


def compute_normal_contrastive_loss(text_features: torch.Tensor,
                                    device: torch.device,
                                    num_classes: int = None) -> torch.Tensor:
    """Encourages text representations of normal vs. anomalies to remain distinct."""
    if num_classes is not None and text_features.shape[0] > num_classes:
        text_features = text_features[:num_classes]
    norm_text = F.normalize(text_features, dim=-1)
    normal_vec = norm_text[0]  # [D]
    anom_vecs = norm_text[1:]  # [C-1, D]
    sim = torch.abs(anom_vecs @ normal_vec)
    return sim.mean()
