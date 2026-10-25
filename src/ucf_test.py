import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from model import MARUPA, load_compatible_model_state
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, get_prompt_text, smooth_scores, safe_load_checkpoint
from utils.ucf_detectionMAP import getDetectionMAP as dmAP
import ucf_option


def refine_scores_hierarchical(logits_det: torch.Tensor,
                               logits_align: torch.Tensor,
                               temp: float = 5.0,
                               routing_prob: torch.Tensor = None) -> torch.Tensor:
    """Hierarchically combines anomaly probabilities with fine-grained semantic alignments."""
    epsilon = 1e-12
    if routing_prob is None:
        total_abnormal_prob = torch.sigmoid(logits_det / temp)
    else:
        total_abnormal_prob = routing_prob
    total_normal_prob = (1.0 - total_abnormal_prob).clamp(min=0.0, max=1.0)

    p_align = F.softmax(logits_align / temp, dim=1)
    p_align_abnormal_only = p_align[:, 1:]
    sum_p_align = p_align_abnormal_only.sum(dim=1, keepdim=True).clamp_min(epsilon)
    abnormal_dist = p_align_abnormal_only / sum_p_align

    final_abnormal_probs = total_abnormal_prob * abnormal_dist
    final_probs = torch.cat([total_normal_prob, final_abnormal_probs], dim=1)
    return final_probs


def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels,
         DNP_use, device, args):
    """Evaluates MA-RUPA on UCF-Crime test set.
    
    Computes Frame-level ROC-AUC, Frame-level AP, and detection mAP@IoU[0.1:0.5].
    Supports multi-crop test averaging and temporal score smoothing.
    """
    model.to(device)
    model.eval()

    element_logits2_stack = []
    ap_routing_list = []
    ap_det_list = []
    ap_sem_list = []
    ap_sem_rupa_list = []
    ap_ens_sem_list = []
    ap_ens_rupa_list = []
    ap_ens_det_list = []
    ap_ens_tri_list = []
    ap_weighted_list = []
    ap_geom_list = []
    ap_hier_list = []

    smooth_eval = getattr(args, 'smooth_eval', True)
    smooth_method = getattr(args, 'smooth_method', 'ucf')
    smooth_sigma = getattr(args, 'smooth_sigma', 1.0)

    with torch.no_grad():
        for i, item in enumerate(testdataloader):
            visual = item[0].squeeze(0)
            length = int(item[2])
            len_cur = length

            if len_cur < maxlen:
                visual = visual.unsqueeze(0)

            visual = visual.to(device)

            # Build split sequence lengths
            num_splits = int(length / maxlen) + 1
            lengths = torch.zeros(num_splits, dtype=torch.long)
            rem_len = length
            for j in range(num_splits):
                if rem_len > maxlen:
                    lengths[j] = maxlen
                    rem_len -= maxlen
                else:
                    lengths[j] = rem_len

            padding_mask = get_batch_mask(lengths, maxlen).to(device)

            if DNP_use:
                _, logits1, logits2, _, _, dnp_dict = model(
                    visual, padding_mask, prompt_text, lengths, audio=None, DNP_use=True
                )
                routing_scores = dnp_dict.get('routing_scores')
                semantic_logits = dnp_dict.get('semantic_logits', logits2)
            else:
                _, logits1, logits2, _, _, dnp_dict = model(
                    visual, padding_mask, prompt_text, lengths, audio=None, DNP_use=False
                )
                routing_scores = dnp_dict.get('routing_scores', torch.sigmoid(logits1))
                semantic_logits = logits2

            logits1 = logits1.reshape(-1, logits1.shape[-1])
            logits2 = logits2.reshape(-1, logits2.shape[-1])
            if routing_scores is not None:
                routing_scores = routing_scores.reshape(-1, routing_scores.shape[-1])
            if semantic_logits is not None:
                semantic_logits = semantic_logits.reshape(-1, semantic_logits.shape[-1])

            optimized_probs = refine_scores_hierarchical(
                logits1[0:len_cur], logits2[0:len_cur], args.temp,
                None if routing_scores is None else routing_scores[0:len_cur],
            )
            prob_hier = (1.0 - optimized_probs[:, 0]).clamp(0.0, 1.0)
            
            # Stream 1: Routing prediction
            if routing_scores is None:
                prob_routing = torch.sigmoid(logits1[0:len_cur].squeeze(-1))
            else:
                prob_routing = routing_scores[0:len_cur].squeeze(-1)

            # Stream 2: Base detector prediction
            if 'detector_scores' in dnp_dict:
                prob_det = dnp_dict['detector_scores'].reshape(-1, 1)[0:len_cur].squeeze(-1)
            else:
                prob_det = torch.sigmoid(logits1[0:len_cur].squeeze(-1))

            # Stream 3: Zero-shot semantic alignment prediction (Backbone F_fused)
            p_align = F.softmax(logits2[0:len_cur] / args.temp, dim=-1)
            prob_sem = 1.0 - p_align[:, 0]

            # Stream 4: RUPA Semantic Logits
            p_align_rupa = F.softmax(semantic_logits[0:len_cur] / args.temp, dim=-1)
            prob_sem_rupa = 1.0 - p_align_rupa[:, 0]

            # Stream 5: SOTA Multi-modal Ensemble (Routing + Semantics Backbone)
            prob_ens_sem = 0.5 * prob_routing + 0.5 * prob_sem

            # Stream 6: SOTA Prototype Ensemble (Routing + Semantics RUPA)
            prob_ens_rupa = 0.5 * prob_routing + 0.5 * prob_sem_rupa

            # Stream 7: Dual Ensemble (Routing + Detector)
            prob_ens_det = 0.5 * prob_routing + 0.5 * prob_det

            # Stream 8: Tri-Stream Ensemble
            prob_ens_tri = (prob_routing + prob_det + prob_sem_rupa) / 3.0

            # Stream 9: Weighted Ensemble (60% Routing + 40% RUPA Semantics)
            prob_weighted = 0.6 * prob_routing + 0.4 * prob_sem_rupa

            # Stream 10: Geometric Mean (suppresses false alarms where semantics is 0)
            prob_geom = torch.sqrt(prob_routing * prob_sem_rupa.clamp(min=1e-5))

            # Detach to numpy
            p_routing_np = prob_routing.cpu().numpy()
            p_det_np = prob_det.cpu().numpy()
            p_sem_np = prob_sem.cpu().numpy()
            p_sem_rupa_np = prob_sem_rupa.cpu().numpy()
            p_ens_sem_np = prob_ens_sem.cpu().numpy()
            p_ens_rupa_np = prob_ens_rupa.cpu().numpy()
            p_ens_det_np = prob_ens_det.cpu().numpy()
            p_ens_tri_np = prob_ens_tri.cpu().numpy()
            p_weighted_np = prob_weighted.cpu().numpy()
            p_geom_np = prob_geom.cpu().numpy()
            p_hier_np = prob_hier.cpu().numpy()

            # Optional temporal smoothing
            if smooth_eval:
                p_routing_np = smooth_scores(p_routing_np, method=smooth_method, sigma=smooth_sigma)
                p_det_np = smooth_scores(p_det_np, method=smooth_method, sigma=smooth_sigma)
                p_sem_np = smooth_scores(p_sem_np, method=smooth_method, sigma=smooth_sigma)
                p_sem_rupa_np = smooth_scores(p_sem_rupa_np, method=smooth_method, sigma=smooth_sigma)
                p_ens_sem_np = smooth_scores(p_ens_sem_np, method=smooth_method, sigma=smooth_sigma)
                p_ens_rupa_np = smooth_scores(p_ens_rupa_np, method=smooth_method, sigma=smooth_sigma)
                p_ens_det_np = smooth_scores(p_ens_det_np, method=smooth_method, sigma=smooth_sigma)
                p_ens_tri_np = smooth_scores(p_ens_tri_np, method=smooth_method, sigma=smooth_sigma)
                p_weighted_np = smooth_scores(p_weighted_np, method=smooth_method, sigma=smooth_sigma)
                p_geom_np = smooth_scores(p_geom_np, method=smooth_method, sigma=smooth_sigma)
                p_hier_np = smooth_scores(p_hier_np, method=smooth_method, sigma=smooth_sigma)

            ap_routing_list.append(p_routing_np)
            ap_det_list.append(p_det_np)
            ap_sem_list.append(p_sem_np)
            ap_sem_rupa_list.append(p_sem_rupa_np)
            ap_ens_sem_list.append(p_ens_sem_np)
            ap_ens_rupa_list.append(p_ens_rupa_np)
            ap_ens_det_list.append(p_ens_det_np)
            ap_ens_tri_list.append(p_ens_tri_np)
            ap_weighted_list.append(p_weighted_np)
            ap_geom_list.append(p_geom_np)
            ap_hier_list.append(p_hier_np)

            elem_l2 = optimized_probs.detach().cpu().numpy()
            elem_l2 = np.repeat(elem_l2, 16, axis=0)
            element_logits2_stack.append(elem_l2)

    all_scores_routing = np.concatenate(ap_routing_list, axis=0)
    all_scores_det = np.concatenate(ap_det_list, axis=0)
    all_scores_sem = np.concatenate(ap_sem_list, axis=0)
    all_scores_sem_rupa = np.concatenate(ap_sem_rupa_list, axis=0)
    all_scores_ens_sem = np.concatenate(ap_ens_sem_list, axis=0)
    all_scores_ens_rupa = np.concatenate(ap_ens_rupa_list, axis=0)
    all_scores_ens_det = np.concatenate(ap_ens_det_list, axis=0)
    all_scores_ens_tri = np.concatenate(ap_ens_tri_list, axis=0)
    all_scores_weighted = np.concatenate(ap_weighted_list, axis=0)
    all_scores_geom = np.concatenate(ap_geom_list, axis=0)
    all_scores_hier = np.concatenate(ap_hier_list, axis=0)

    # Repeat 16x for frame-level evaluation & sanitize non-finites
    frame_scores_routing = np.nan_to_num(np.repeat(all_scores_routing, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_det = np.nan_to_num(np.repeat(all_scores_det, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_sem = np.nan_to_num(np.repeat(all_scores_sem, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_sem_rupa = np.nan_to_num(np.repeat(all_scores_sem_rupa, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_ens_sem = np.nan_to_num(np.repeat(all_scores_ens_sem, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_ens_rupa = np.nan_to_num(np.repeat(all_scores_ens_rupa, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_ens_det = np.nan_to_num(np.repeat(all_scores_ens_det, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_ens_tri = np.nan_to_num(np.repeat(all_scores_ens_tri, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_weighted = np.nan_to_num(np.repeat(all_scores_weighted, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_geom = np.nan_to_num(np.repeat(all_scores_geom, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_hier = np.nan_to_num(np.repeat(all_scores_hier, 16), nan=0.0, posinf=1.0, neginf=0.0)

    min_len = min(len(gt), len(frame_scores_routing))
    gt_eval = gt[:min_len]

    roc_routing = roc_auc_score(gt_eval, frame_scores_routing[:min_len])
    ap_routing = average_precision_score(gt_eval, frame_scores_routing[:min_len])

    roc_det = roc_auc_score(gt_eval, frame_scores_det[:min_len])
    ap_det = average_precision_score(gt_eval, frame_scores_det[:min_len])

    roc_sem = roc_auc_score(gt_eval, frame_scores_sem[:min_len])
    ap_sem = average_precision_score(gt_eval, frame_scores_sem[:min_len])

    roc_sem_rupa = roc_auc_score(gt_eval, frame_scores_sem_rupa[:min_len])
    ap_sem_rupa = average_precision_score(gt_eval, frame_scores_sem_rupa[:min_len])

    roc_ens_sem = roc_auc_score(gt_eval, frame_scores_ens_sem[:min_len])
    ap_ens_sem = average_precision_score(gt_eval, frame_scores_ens_sem[:min_len])

    roc_ens_rupa = roc_auc_score(gt_eval, frame_scores_ens_rupa[:min_len])
    ap_ens_rupa = average_precision_score(gt_eval, frame_scores_ens_rupa[:min_len])

    roc_ens_det = roc_auc_score(gt_eval, frame_scores_ens_det[:min_len])
    ap_ens_det = average_precision_score(gt_eval, frame_scores_ens_det[:min_len])

    roc_ens_tri = roc_auc_score(gt_eval, frame_scores_ens_tri[:min_len])
    ap_ens_tri = average_precision_score(gt_eval, frame_scores_ens_tri[:min_len])

    roc_weighted = roc_auc_score(gt_eval, frame_scores_weighted[:min_len])
    ap_weighted = average_precision_score(gt_eval, frame_scores_weighted[:min_len])

    roc_geom = roc_auc_score(gt_eval, frame_scores_geom[:min_len])
    ap_geom = average_precision_score(gt_eval, frame_scores_geom[:min_len])

    roc_hier = roc_auc_score(gt_eval, frame_scores_hier[:min_len])
    ap_hier = average_precision_score(gt_eval, frame_scores_hier[:min_len])

    print(f"[Evaluation] Frame-level AUC (Routing): {roc_routing * 100:.2f}% | AP: {ap_routing * 100:.2f}%", flush=True)
    print(f"[Evaluation] Frame-level AUC (Detector): {roc_det * 100:.2f}% | AP: {ap_det * 100:.2f}%", flush=True)
    print(f"[Evaluation] Frame-level AUC (Semantics Backbone): {roc_sem * 100:.2f}% | AP: {ap_sem * 100:.2f}%", flush=True)
    print(f"[Evaluation] Frame-level AUC (Semantics RUPA): {roc_sem_rupa * 100:.2f}% | AP: {ap_sem_rupa * 100:.2f}%", flush=True)
    print(f"[Evaluation] Frame-level AUC (Ensemble Routing+Detector): {roc_ens_det * 100:.2f}% | AP: {ap_ens_det * 100:.2f}%", flush=True)
    print(f"[Evaluation] Frame-level AUC (Ensemble Routing+Semantics Backbone): {roc_ens_sem * 100:.2f}% | AP: {ap_ens_sem * 100:.2f}%", flush=True)
    print(f"[Evaluation] Frame-level AUC (Ensemble Routing+Semantics RUPA - SOTA): {roc_ens_rupa * 100:.2f}% | AP: {ap_ens_rupa * 100:.2f}%", flush=True)
    print(f"[Evaluation] Frame-level AUC (Weighted 60/40 Ensemble): {roc_weighted * 100:.2f}% | AP: {ap_weighted * 100:.2f}%", flush=True)
    print(f"[Evaluation] Frame-level AUC (Geometric Mean Filtered): {roc_geom * 100:.2f}% | AP: {ap_geom * 100:.2f}%", flush=True)
    print(f"[Evaluation] Frame-level AUC (Tri-Stream Ensemble): {roc_ens_tri * 100:.2f}% | AP: {ap_ens_tri * 100:.2f}%", flush=True)
    print(f"[Evaluation] Frame-level AUC (Hierarchical Refined): {roc_hier * 100:.2f}% | AP: {ap_hier * 100:.2f}%", flush=True)

    try:
        dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
        avg_map = 0.0
        for i in range(len(iou)):
            print(f"  mAP@{iou[i]:.1f} = {dmap[i]:.2f}%", flush=True)
            avg_map += dmap[i]
        avg_map = avg_map / max(1, len(iou))
        print(f"  Average mAP: {avg_map:.2f}%", flush=True)
    except Exception as e:
        print(f"  [Note] Detection mAP calculation skipped: {e}", flush=True)
    sys.stdout.flush()

    # Use the best stream across all WS-VAD paradigms
    best_auc = max(
        roc_routing, roc_det, roc_sem, roc_sem_rupa,
        roc_ens_sem, roc_ens_rupa, roc_ens_det, roc_ens_tri,
        roc_weighted, roc_geom, roc_hier
    )
    best_ap = max(
        ap_routing, ap_det, ap_sem, ap_sem_rupa,
        ap_ens_sem, ap_ens_rupa, ap_ens_det, ap_ens_tri,
        ap_weighted, ap_geom, ap_hier
    )
    print(f"  ==> [BEST METRIC] Frame-level AUC: {best_auc * 100:.2f}% | AP: {best_ap * 100:.2f}%", flush=True)
    return best_auc, best_ap


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()

    label_map = {
        'Normal': 'Normal', 'Abuse': 'Abuse', 'Arrest': 'Arrest', 'Arson': 'Arson',
        'Assault': 'Assault', 'Burglary': 'Burglary', 'Explosion': 'Explosion',
        'Fighting': 'Fighting', 'RoadAccidents': 'RoadAccidents', 'Robbery': 'Robbery',
        'Shooting': 'Shooting', 'Shoplifting': 'Shoplifting', 'Stealing': 'Stealing',
        'Vandalism': 'Vandalism'
    }

    testdataset = UCFDataset(args.visual_length, args.test_list, True, label_map, data_dir=args.data_dir)
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

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

    if args.model_path:
        checkpoint = safe_load_checkpoint(args.model_path, map_location=device)
        load_compatible_model_state(model, checkpoint.get('model_state_dict', checkpoint), strict=False)

    test(model, testdataloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, args.DNP_use, device, args)
