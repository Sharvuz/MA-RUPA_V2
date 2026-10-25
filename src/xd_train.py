"""MA-RUPA End-to-End Training Script for XD-Violence.

Trains MA-RUPA from scratch in a single stage with multimodal audio-visual fusion
and curriculum warm-up to prevent gradient collapse.
"""

import math
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import MARUPA, load_compatible_model_state
from xd_test import test
from utils.dataset import XDDataset
from utils.tools import get_batch_label, get_prompt_text, xd_collate_fn, xd_test_collate_fn, safe_load_checkpoint
from utils.losses import (
    compute_stochastic_topk_mil_loss,
    compute_shock_tv_loss,
    compute_haar_wavelet_loss,
    compute_geodesic_halo_loss,
    compute_clasm_loss,
    compute_bkg_loss,
    compute_consistency_loss,
    compute_event_loss,
    compute_residual_event_loss,
    compute_normal_contrastive_loss,
)
import xd_option


def compute_topk_mil_loss(scores: torch.Tensor, binary_labels: torch.Tensor,
                         lengths: torch.Tensor, device: torch.device,
                         topk_ratio: int = 16, label_smoothing: float = 0.0):
    """Computes Top-K Multiple Instance Learning (MIL) Binary Cross-Entropy loss with optional label smoothing."""
    instance_scores = []
    binary_labels = binary_labels.to(device)
    scores_flat = scores.squeeze(-1)  # [B, T]

    for i in range(scores.shape[0]):
        l_i = int(lengths[i].item())
        k_count = max(1, int(l_i / topk_ratio) + 1)
        valid_scores = scores_flat[i, :l_i]
        topk_vals, _ = torch.topk(valid_scores, k=min(k_count, l_i), largest=True)
        instance_scores.append(topk_vals.mean().view(1))

    instance_scores = torch.cat(instance_scores, dim=0)
    targets = binary_labels.float()
    if label_smoothing > 0.0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing

    loss = F.binary_cross_entropy(instance_scores.clamp(1e-6, 1.0 - 1e-6), targets)
    return loss


def compute_temporal_smoothness_loss(scores: torch.Tensor, lengths: torch.Tensor,
                                     lambda_sparse: float = 8e-5):
    """Computes temporal total variation smoothness and snippet sparsity loss."""
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


def compute_residual_event_loss(residual_logits: torch.Tensor, labels: torch.Tensor,
                                lengths: torch.Tensor, device: torch.device):
    """Aligns anomaly residual features with fine-grained anomaly text prompts."""
    # For XD-Violence: index 0 is Normal ('A'); indices 1..6 are anomaly classes
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


def compute_clasm_loss(logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor,
                       device: torch.device, k_ratio: int = 16, epsilon: float = 0.05):
    """Snippet-level multi-class MIL cross-entropy loss against fine-grained categories (Normal + 6 violence classes)."""
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


def compute_bkg_loss(logits4: torch.Tensor, device: torch.device, num_classes: int = 7, epsilon: float = 0.05):
    """Background-centric loss forcing reconstructed prototypes to classify as class 0 (Normal)."""
    nor_logits = logits4.squeeze(1)  # [B, C]
    target = torch.full_like(nor_logits, epsilon / num_classes, device=device)
    target[:, 0] = 1.0 - epsilon + (epsilon / num_classes)
    loss = -torch.mean(torch.sum(target * F.log_softmax(nor_logits, dim=1), dim=1))
    return loss


def compute_consistency_loss(p_routing: torch.Tensor, original_features: torch.Tensor,
                             reconstructed_features: torch.Tensor, lengths: torch.Tensor,
                             device: torch.device):
    """Enforces consistency between detector routing anomaly score and prototype reconstruction error."""
    cos_sim = F.cosine_similarity(original_features, reconstructed_features, dim=-1)  # [B, T]
    recon_error = ((1.0 - cos_sim) / 2.0).clamp(0.0, 1.0)
    pred_score = p_routing.squeeze(-1)  # [B, T]

    B, T = pred_score.shape
    mask = torch.arange(T, device=device)[None, :] < lengths.to(device)[:, None]

    if not torch.any(mask):
        return torch.tensor(0.0, device=device)

    return F.mse_loss(pred_score[mask], recon_error[mask])


def compute_event_loss(logits3: torch.Tensor, labels: torch.Tensor, device: torch.device, epsilon: float = 0.05):
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


def compute_normal_contrastive_loss(text_features: torch.Tensor, device: torch.device, num_classes: int = None):
    """Encourages text representations of normal vs. anomalies to remain distinct."""
    if num_classes is not None and text_features.shape[0] > num_classes:
        text_features = text_features[:num_classes]
    norm_text = F.normalize(text_features, dim=-1)
    normal_vec = norm_text[0]  # [D]
    anom_vecs = norm_text[1:]  # [C-1, D]
    sim = torch.abs(anom_vecs @ normal_vec)
    return sim.mean()


def train(args):
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MA-RUPA] Training XD-Violence on device: {device}", flush=True)

    os.makedirs(os.path.dirname(args.model_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.checkpoint_path) or '.', exist_ok=True)

    label_map = {
        'A': 'normal', 'B1': 'fighting', 'B2': 'shooting',
        'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'
    }

    # Datasets & Loaders
    train_dataset = XDDataset(
        args.visual_length, args.train_list, False, label_map,
        audio_file_path=args.audio_train_list, audio_dir=args.audio_dir,
        data_dir=args.data_dir, no_audio=args.no_audio
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=xd_collate_fn, drop_last=True, pin_memory=True
    )

    test_dataset = XDDataset(
        args.visual_length, args.test_list, True, label_map,
        audio_file_path=args.audio_test_list, audio_dir=args.audio_dir,
        data_dir=args.data_dir, no_audio=args.no_audio
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=xd_test_collate_fn)

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    # Auto-detect audio feature dimension from dataset if available
    detected_audio_dim = None
    for idx in range(min(50, len(train_dataset))):
        item_tuple = train_dataset[idx]
        sample_audio = item_tuple[3] if len(item_tuple) >= 4 else item_tuple[-1]
        if sample_audio is not None and isinstance(sample_audio, torch.Tensor):
            detected_audio_dim = sample_audio.shape[-1]
            break
    if detected_audio_dim is not None and detected_audio_dim != args.audio_dim:
        print(f"[Auto-Detect] Audio feature dimension on disk is {detected_audio_dim}-D (command-line was {args.audio_dim}-D). Auto-adjusting args.audio_dim to {detected_audio_dim}!", flush=True)
        args.audio_dim = detected_audio_dim

    # Initialize model
    model = MARUPA(
        num_class=args.classes_num,
        embed_dim=args.embed_dim,
        visual_length=args.visual_length,
        visual_width=args.visual_width,
        visual_head=args.visual_head,
        visual_layers=args.visual_layers,
        attn_window=args.attn_window,
        args=args,
        device=device
    )
    model.to(device)

    # Multi-GPU support (e.g. Kaggle GPU T4 x 2)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        print(f"[Multi-GPU] Detected {torch.cuda.device_count()} GPUs ({', '.join(gpu_names)}). Activating DataParallel for T4x2!", flush=True)
        model = torch.nn.DataParallel(model)

    # Single-stage End-to-End AdamW optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    iters_per_epoch = len(train_loader)
    total_iters = max(1, args.max_epoch * iters_per_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_iters, eta_min=args.lr * 0.01
    )

    best_ap = 0.0
    best_auc = 0.0
    start_epoch = 0

    if args.use_checkpoint and os.path.exists(args.checkpoint_path):
        print(f"Resuming from checkpoint: {args.checkpoint_path}", flush=True)
        cp = safe_load_checkpoint(args.checkpoint_path, map_location=device)
        load_compatible_model_state(model, cp['model_state_dict'], strict=False)
        optimizer.load_state_dict(cp['optimizer_state_dict'])
        scheduler.load_state_dict(cp['scheduler_state_dict'])
        start_epoch = cp['epoch'] + 1
        best_ap = cp.get('best_ap', 0.0)
        best_auc = cp.get('best_auc', 0.0)

    print("\n" + "=" * 70)
    print(f"[START] STARTING END-TO-END TRAINING FOR XD-VIOLENCE: {args.max_epoch} EPOCHS")
    print("=" * 70, flush=True)

    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        epoch_idx = epoch + 1
        # Smooth Cosine Curriculum Schedule capped strictly at omega_max = 0.35
        curriculum_eps = float(getattr(args, 'curriculum_epochs', 5.0))
        if epoch_idx <= curriculum_eps:
            progress = (epoch_idx - 1) / max(1.0, curriculum_eps - 1)
            omega = 0.05 + 0.30 * 0.5 * (1.0 - math.cos(math.pi * progress))
        else:
            omega = 0.35

        loss_accum = {
            'mil': 0.0, 'aux': 0.0, 'clasm': 0.0, 'bkg': 0.0, 'consist': 0.0,
            'shock': 0.0, 'wavelet': 0.0, 'halo': 0.0, 'orth': 0.0, 'residual': 0.0,
            'gather': 0.0, 'total': 0.0
        }

        for step_idx, batch in enumerate(train_loader):
            visual_batch, text_labels, lengths_batch, audio_batch = batch
            visual_batch = visual_batch.to(device)
            lengths_batch = lengths_batch.to(device)
            if audio_batch is not None:
                audio_batch = audio_batch.to(device)

            # Binary labels: 0 for normal ('A'), 1 for violence/anomaly
            binary_labels = torch.tensor([
                0.0 if lbl == 'A' or lbl.startswith('A-') else 1.0
                for lbl in text_labels
            ], device=device)

            # Fine-grained categorical one-hot tensor
            one_hot_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            # Model Forward Pass (Multimodal Audio-Visual if audio present, else Motion Residuals)
            text_features, logits1, logits2, logits3, logits4, dnp_dict = model(
                visual=visual_batch,
                padding_mask=None,
                text=prompt_text,
                lengths=lengths_batch,
                audio=audio_batch,
                DNP_use=args.DNP_use
            )

            p_routing = dnp_dict['routing_scores']
            s_aux = dnp_dict['aux_scores']
            s_det = dnp_dict['detector_scores']
            orig_features = dnp_dict.get('original_features', visual_batch)

            # 1. Primary & Auxiliary MIL Losses with Stochastic Peak-Drop
            topk_r = getattr(args, 'topk_ratio', 16)
            lbl_sm = getattr(args, 'label_smoothing', 0.02)
            loss_mil_routing = compute_stochastic_topk_mil_loss(
                p_routing, binary_labels, lengths_batch, device,
                topk_ratio=topk_r, label_smoothing=lbl_sm, training=True, p_drop=0.30, drop_ratio=0.20
            ).mean()
            loss_mil_aux = compute_stochastic_topk_mil_loss(
                s_aux, binary_labels, lengths_batch, device,
                topk_ratio=topk_r, label_smoothing=lbl_sm, training=True, p_drop=0.30, drop_ratio=0.20
            ).mean()
            loss_mil_det = compute_stochastic_topk_mil_loss(
                s_det, binary_labels, lengths_batch, device,
                topk_ratio=topk_r, label_smoothing=lbl_sm, training=True, p_drop=0.30, drop_ratio=0.20
            ).mean()

            # 2. Perona-Malik Shock-Preserving Smoothness (lambda=0.002) & 1D Haar Wavelet Contrast (lambda=0.03)
            loss_shock = compute_shock_tv_loss(p_routing, orig_features, lengths_batch).mean()
            loss_wavelet = compute_haar_wavelet_loss(p_routing, binary_labels, lengths_batch).mean()

            # 3. One-Sided Geodesic Halo Expansion Loss (lambda=0.12)
            loss_halo = compute_geodesic_halo_loss(p_routing, orig_features, binary_labels, lengths_batch).mean()

            # 4. Multi-class Semantic Alignment & Consistency Losses
            loss_clasm = compute_clasm_loss(logits2, one_hot_labels, lengths_batch, device).mean()
            loss_bkg = compute_bkg_loss(logits4, device, num_classes=len(prompt_text)).mean()

            if 'original_features' in dnp_dict and 'reconstructed_features' in dnp_dict:
                loss_consistency = compute_consistency_loss(
                    p_routing, dnp_dict['original_features'], dnp_dict['reconstructed_features'], lengths_batch, device
                ).mean()
            else:
                loss_consistency = torch.tensor(0.0, device=device)

            loss_event = compute_event_loss(logits3, one_hot_labels, device).mean()
            loss_dnp_normal = dnp_dict.get('dnp_normal_loss', torch.tensor(0.0, device=device)).mean()

            # 5. RUPA Prototype & Orthogonality Losses (modulated by curriculum ramp-up omega(e) <= 0.35)
            loss_gather = dnp_dict.get('g_loss', torch.tensor(0.0, device=device)).mean()
            loss_orth = dnp_dict.get('orth_loss', torch.tensor(0.0, device=device)).mean()

            if 'residual_semantic_logits' in dnp_dict:
                loss_residual = compute_residual_event_loss(
                    dnp_dict['residual_semantic_logits'], one_hot_labels, lengths_batch, device
                ).mean()
            else:
                loss_residual = torch.tensor(0.0, device=device)

            gate_val = dnp_dict.get('gate', torch.tensor(0.0, device=device))
            loss_gate = (gate_val ** 2).mean()

            # Text Orthogonality Regularization
            loss_text_orth = compute_normal_contrastive_loss(
                text_features, device, num_classes=len(prompt_text)
            ).mean()

            # Total Loss Computation (guaranteed 0-D scalar for backward())
            total_loss = (
                loss_mil_routing
                + 0.5 * loss_mil_det
                + args.lambda_aux * loss_mil_aux
                + 0.002 * loss_shock
                + 0.03 * loss_wavelet
                + 0.12 * loss_halo
                + getattr(args, 'loss2_weight', 1.0) * loss_clasm
                + getattr(args, 'lambda_bkg', 0.5) * loss_bkg
                + getattr(args, 'lambda_consistency', 1.0) * loss_consistency
                + 0.1 * loss_text_orth
                + omega * (
                    args.lambda_gather * loss_gather
                    + args.lambda_orth * loss_orth
                    + args.lambda_residual * loss_residual
                    + 0.5 * loss_event
                    + 0.2 * loss_dnp_normal
                    + args.lambda_gate * loss_gate
                )
            ).mean()

            optimizer.zero_grad()
            total_loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip)

            optimizer.step()
            scheduler.step()

            # Accumulate logs
            loss_accum['mil'] += loss_mil_routing.item()
            loss_accum['aux'] += loss_mil_aux.item()
            loss_accum['clasm'] += loss_clasm.item()
            loss_accum['bkg'] += loss_bkg.item()
            loss_accum['consist'] += loss_consistency.item()
            loss_accum['shock'] += loss_shock.item()
            loss_accum['wavelet'] += loss_wavelet.item()
            loss_accum['halo'] += loss_halo.item()
            loss_accum['orth'] += loss_orth.item()
            loss_accum['residual'] += loss_residual.item()
            loss_accum['gather'] += loss_gather.item()
            loss_accum['total'] += total_loss.item()

            total_batches = len(train_loader)
            if (step_idx + 1) % 25 == 0 or (step_idx + 1) == total_batches:
                current_samples = min((step_idx + 1) * args.batch_size, len(train_dataset))
                percent = (current_samples / len(train_dataset)) * 100
                print(f"  [Epoch {epoch_idx}/{args.max_epoch} Step {current_samples}/{len(train_dataset)} ({percent:.1f}%)] "
                      f"Total: {total_loss.item():.4f} | MIL: {loss_mil_routing.item():.4f} "
                      f"CLASM: {loss_clasm.item():.4f} | Consist: {loss_consistency.item():.4f} | Aux: {loss_mil_aux.item():.4f} | omega: {omega:.2f}", flush=True)

        # End of Epoch Reporting
        avg_total = loss_accum['total'] / iters_per_epoch
        avg_mil = loss_accum['mil'] / iters_per_epoch
        avg_clasm = loss_accum['clasm'] / iters_per_epoch
        avg_consist = loss_accum['consist'] / iters_per_epoch
        avg_aux = loss_accum['aux'] / iters_per_epoch
        avg_gather = loss_accum.get('gather', 0.0) / iters_per_epoch
        avg_shock = loss_accum.get('shock', 0.0) / iters_per_epoch
        avg_wavelet = loss_accum.get('wavelet', 0.0) / iters_per_epoch
        avg_halo = loss_accum.get('halo', 0.0) / iters_per_epoch
        avg_orth = loss_accum.get('orth', 0.0) / iters_per_epoch

        print(f"\n--- Epoch {epoch_idx}/{args.max_epoch} Summary ---")
        print(f"Loss Total: {avg_total:.4f} | MIL: {avg_mil:.4f} | CLASM: {avg_clasm:.4f} | Consist: {avg_consist:.4f} | Aux: {avg_aux:.4f} | Gather: {avg_gather:.4f} | Shock: {avg_shock:.4f} | Wave: {avg_wavelet:.4f} | Halo: {avg_halo:.4f} | Orth: {avg_orth:.4f} | Omega: {omega:.2f}")

        # Evaluate on Test Set
        eval_model = model.module if hasattr(model, 'module') else model
        auc_val, ap_val, map_val = test(
            eval_model, test_loader, args.visual_length, prompt_text, gt, gtsegments,
            gtlabels, args.DNP_use, args, device
        )

        is_best = False
        saved_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        if ap_val > best_ap:
            best_ap = ap_val
            best_auc = auc_val
            is_best = True
            torch.save({
                'epoch': epoch,
                'model_state_dict': saved_state,
                'ap': best_ap,
                'auc': best_auc,
                'args': vars(args)
            }, args.model_path)
            print(f"[BEST MODEL] Best AP: {best_ap * 100:.2f}% | AUC: {best_auc * 100:.2f}% saved to {args.model_path}", flush=True)

        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': saved_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'ap': ap_val,
            'auc': auc_val,
            'best_ap': best_ap,
            'best_auc': best_auc
        }, args.checkpoint_path)

    print("\n" + "=" * 70)
    print(f"[COMPLETE] XD-VIOLENCE TRAINING COMPLETE! FINAL BEST AP: {best_ap * 100:.2f}% | AUC: {best_auc * 100:.2f}%")
    print("=" * 70 + "\n", flush=True)


if __name__ == '__main__':
    args = xd_option.parser.parse_args()
    train(args)
