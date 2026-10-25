import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError('Expected a boolean value.')


parser = argparse.ArgumentParser(description='MA-RUPA: Modality-Adaptive Multiscale Prototype Alignment Network (XD-Violence)')
parser.add_argument('--seed', default=234, type=int, help='Random seed for reproducibility')

# Architecture Hyperparameters
parser.add_argument('--embed-dim', default=512, type=int, help='Feature embedding dimension')
parser.add_argument('--visual-length', default=256, type=int, help='Maximum temporal snippets per video')
parser.add_argument('--visual-width', default=512, type=int, help='Visual feature width')
parser.add_argument('--visual-head', default=4, type=int, help='Attention heads in backbone & cross-attention')
parser.add_argument('--visual-layers', default=2, type=int, help='Number of transformer layers in Branch A')
parser.add_argument('--attn-window', default=64, type=int, help='Window size for local windowed attention (Branch A)')
parser.add_argument('--classes-num', default=7, type=int, help='Number of classes for XD-Violence (6 anomalies + 1 normal)')
parser.add_argument('--mlp-dropout', default=0.1, type=float, help='Dropout rate for MLPs')

# Audio Multimodal Branch Hyperparameters
parser.add_argument('--audio-dim', default=128, type=int, help='Audio feature dimension (128 for VGGish, 512 for CLAP/AST)')
parser.add_argument('--audio-train-list', default=None, type=str, help='Optional CSV mapping visual clips to audio features')
parser.add_argument('--audio-test-list', default=None, type=str, help='Optional CSV mapping test visual clips to audio features')
parser.add_argument('--audio-dir', default=None, type=str, help='Optional directory containing audio .npy feature files')

# RUPA Normality & Prototype Hyperparameters
parser.add_argument('--num_prototypes', type=int, default=16, help='Number of Dynamic Normal Prototypes (M=16)')
parser.add_argument('--decoder_depth', type=int, default=4, help='Prototype reconstruction decoder depth')
parser.add_argument('--sampling', choices=['topk', 'otsu'], default='topk', help='Pseudo-normal segment selection method')
parser.add_argument('--normal_selection_ratio', type=float, default=0.8, help='Selection ratio for top-k normal mining')
parser.add_argument('--min_normal_frames', type=int, default=4, help='Minimum normal frames to select per video')
parser.add_argument('--max_normal_ratio', type=float, default=0.8, help='Maximum normal frames ratio')
parser.add_argument('--DNP_use', default=True, type=str2bool, help='Whether to use Dynamic Normal Prototypes')

# Adapter & Temperature
parser.add_argument('--text_adapt_until', default=1, type=int, help='CLIP text adapter depth')
parser.add_argument('--t_w', default=0.6, type=float, help='Text adapter residual interpolation weight')
parser.add_argument('--temp', default=2.0, type=float, help='Logit scaling temperature for evaluation refinement')

# Ablation Flags
parser.add_argument('--no-mdtp', action='store_true', default=False, help='Ablation: disable Multi-Scale Dilated Temporal Pyramid')
parser.add_argument('--no-audio', action='store_true', default=False, help='Ablation: disable audio branch (falls back to motion residuals)')
parser.add_argument('--no-rupa', action='store_true', default=False, help='Ablation: disable RUPA prototype reconstruction branch')
parser.add_argument('--no-safegate', action='store_true', default=False, help='Ablation: disable dynamic uncertainty Safe Gate')
parser.add_argument('--routing-mode', choices=['fixed', 'safe_gate'], default='safe_gate', type=str)
parser.add_argument('--routing-gate-hidden', default=16, type=int, help='Hidden dimension of dynamic safe gate MLP')

# Prior weights for XD-Violence (with Audio)
parser.add_argument('--routing-det-weight', default=0.40, type=float, help='Prior weight for detector stream')
parser.add_argument('--routing-rec-weight', default=0.25, type=float, help='Prior weight for reconstruction stream')
parser.add_argument('--routing-sem-weight', default=0.15, type=float, help='Prior weight for semantic stream')
parser.add_argument('--routing-aux-weight', default=0.20, type=float, help='Prior weight for auxiliary audio-visual stream')

# Training & Optimization
parser.add_argument('--max-epoch', default=12, type=int, help='Total training epochs')
parser.add_argument('--batch-size', default=96, type=int, help='Batch size for training')
parser.add_argument('--lr', type=float, default=1.2e-4, help='Learning rate for End-to-End AdamW optimizer')
parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for regularization')
parser.add_argument('--curriculum-epochs', type=float, default=3.0, help='Epochs to ramp up RUPA losses: omega(e) = min(1, e/curriculum_epochs)')
parser.add_argument('--omega-min', type=float, default=0.05, help='Initial minimum weight for prototype curriculum (default 0.05)')
parser.add_argument('--omega-max', type=float, default=1.0, help='Maximum weight for prototype curriculum (default 1.0)')
parser.add_argument('--topk-ratio', type=int, default=16, help='Divisor for top-k snippet selection: k = max(1, L // topk_ratio)')
parser.add_argument('--label-smoothing', type=float, default=0.02, help='Label smoothing factor for binary MIL loss')

# Loss Weighting
parser.add_argument('--lambda-aux', type=float, default=0.8, help='Weight for auxiliary multimodal MIL loss')
parser.add_argument('--lambda-smooth', type=float, default=0.3, help='Weight for temporal smoothness & sparsity loss')
parser.add_argument('--lambda-orth', type=float, default=0.1, help='Weight for prototype orthogonality loss')
parser.add_argument('--lambda-gather', type=float, default=1.0, help='Weight for DNP gather loss')
parser.add_argument('--lambda-residual', type=float, default=1.0, help='Weight for residual fine-grained event alignment loss')
parser.add_argument('--lambda-gate', type=float, default=0.01, help='Weight for safe gate regularization loss')
parser.add_argument('--loss2_weight', type=float, default=1.5, help='Weight for snippet classification loss')
parser.add_argument('--lambda-consistency', type=float, default=1.0, help='Weight for consistency loss between detector and reconstruction')
parser.add_argument('--lambda-bkg', type=float, default=0.5, help='Weight for background normal loss')

# Data & Paths
parser.add_argument('--train-list', default='list/xd_CLIP_rgb.csv', help='Path to training feature CSV list')
parser.add_argument('--test-list', default='list/xd_CLIP_rgbtest.csv', help='Path to test feature CSV list')
parser.add_argument('--gt-path', default='list/gt.npy', help='Path to ground truth frame labels')
parser.add_argument('--gt-segment-path', default='list/gt_segment.npy', help='Path to ground truth segments')
parser.add_argument('--gt-label-path', default='list/gt_label.npy', help='Path to ground truth video labels')
parser.add_argument('--data-dir', default=None, type=str, help='Base directory to resolve/remap feature file paths')
parser.add_argument('--model-path', default='model/model_xd.pth', help='Path to save best model checkpoint')
parser.add_argument('--checkpoint-path', default='model/checkpoint_xd.pth', help='Path to save latest checkpoint')
parser.add_argument('--use-checkpoint', default=False, type=str2bool, help='Resume from checkpoint')

# Post-processing & Multi-crop Test Evaluation
parser.add_argument('--smooth-eval', default=True, type=str2bool, help='Apply temporal score smoothing during evaluation')
parser.add_argument('--smooth-method', choices=['gaussian', 'savgol'], default='gaussian', help='Temporal smoothing filter method')
parser.add_argument('--smooth-sigma', default=1.2, type=float, help='Gaussian filter sigma for score smoothing')
parser.add_argument('--eval-interval', default=2000, type=int, help='Step interval for mid-epoch evaluation')
parser.add_argument('--num-workers', default=2, type=int, help='DataLoader workers')
parser.add_argument('--grad-clip', default=1.0, type=float, help='Max gradient norm for clipping')
