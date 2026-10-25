# PROJECT CONTEXT: MA-RUPA (Modality-Adaptive Multiscale Prototype Alignment Network)

> **Mục đích tài liệu:** Lưu trữ toàn bộ ngữ cảnh kỹ thuật, kiến trúc, cấu hình siêu tham số, trạng thái huấn luyện hiện tại và lộ trình phát triển của dự án **MA-RUPA** để AI Agent (Antigravity/Cursor/Claude) trên bất kỳ máy tính hoặc phiên làm việc nào có thể đọc hiểu và tiếp tục công việc ngay lập tức mà không bị mất dấu.

---

## 1. Tổng quan Dự án (Project Overview)

- **Tên đề tài/Dự án:** MA-RUPA: Modality-Adaptive Multiscale Prototype Alignment Network for Weakly Supervised Video Anomaly Detection (WSVAD).
- **Mục tiêu Benchmark (Chuẩn đầu ra tạp chí Q1 / KLTN xuất sắc):**
  - **UCF-Crime:** Đạt Frame-level ROC-AUC $\ge 89.0\% - 90.0\%$.
  - **XD-Violence:** Đạt Frame-level Average Precision (AP) $\ge 86.0\% - 87.5\%$ (với Batch Size = 96 trên Dual GPU T4x2).
- **GitHub Repository:** `https://github.com/Sharvuz/MA-RUPA_V2.git` (Branch: `main`).
- **Môi trường huấn luyện:** Kaggle GPU Dual Tesla T4 (T4x2 - $2 \times 15$ GB VRAM).
- **Môi trường phát triển cục bộ:** Windows 11, Python 3.10+ Virtual Environment (`.venv`), PyTorch CUDA.

---

## 2. Kiến trúc Kỹ thuật Cốt lõi (Architecture Highlights)

Mô hình MA-RUPA kết hợp 4 luồng thông tin (Quad-Stream Fusion) với cơ chế phân rã và điều phối thích nghi:

1. **Backbone Đa tỉ lệ (MDTP - Multi-Scale Dilated Temporal Pyramid):**
   - Trích xuất đặc trưng thời gian đa tầng với các dilated convolutions ($r \in \{1, 2, 4\}$).
   - Tích hợp **Padded Sequence Window Attention NaN Guard**: Ngăn chặn hiện tượng mẫu padding gây ra $0/0 \rightarrow \text{NaN}$ trong attention matrix.

2. **Nhánh Đa phương thức Thích nghi (Modality-Adaptive Branch):**
   - **XD-Violence (Audio-Visual):** Tự động liên kết Cross-Attention giữa Visual features ($512$-D) và Audio features ($128$-D VGGish / $512$-D CLAP). Cơ chế *split count mismatch recovery* tự động điều chỉnh nếu số frame visual và audio lệch nhau.
   - **UCF-Crime (Motion Residuals):** Tự động chuyển đổi khi không có audio sang tính toán đạo hàm chuyển động thời gian $\Delta V_t = V_{t} - V_{t-1}$ để bắt bạo lực chuyển động đột ngột.

3. **Nguyên mẫu Bình thường Động (DNP - Dynamic Normal Prototypes / RUPA):**
   - Khai phá trực tuyến các đoạn video giả định bình thường (Pseudo-Normal Snippets) thông qua Top-K hoặc Otsu thresholding.
   - Học $M=16$ Prototypes đại diện cho không gian "bình thường".
   - Tính toán sai số tái tạo (Reconstruction Error) $S_{\text{rec}} = 1 - \cos(\mathbf{F}_{\text{orig}}, \mathbf{F}_{\text{recon}})$. Càng bất thường thì sai số tái tạo càng cao.

4. **Bộ phân phối Động Safe Gate (QuadSafeGate):**
   - Thay thế cơ chế gán trọng số cố định. Mạng MLP ước lượng độ bất định (uncertainty) của từng luồng và phân phối trọng số động:
     $$S_{\text{routing}} = w_{\text{det}} S_{\text{det}} + w_{\text{rec}} S_{\text{rec}} + w_{\text{sem}} S_{\text{sem}} + w_{\text{aux}} S_{\text{aux}}$$

5. **Bộ căn chỉnh Ngữ nghĩa Đa nhãn (Multi-class Semantic Alignment - CLASM):**
   - Dùng Text Prompts của CLIP mã hóa tên các hành vi bạo lực (14 lớp cho UCF-Crime, 7 lớp cho XD-Violence bao gồm Normal).
   - Huấn luyện căn chỉnh đa nhãn nhằm tránh việc mô hình chỉ học phân loại nhị phân thô sơ.

---

## 3. Hệ thống Hàm mất mát (Loss Formulation)

Tổng hàm mất mát đầu cuối (End-to-End Scalar Loss, tương thích `torch.nn.DataParallel`):

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{mil}}^{\text{routing}} + 0.5 \mathcal{L}_{\text{mil}}^{\text{det}} + \lambda_{\text{aux}} \mathcal{L}_{\text{mil}}^{\text{aux}} + \lambda_{\text{clasm}} \mathcal{L}_{\text{clasm}} + \lambda_{\text{bkg}} \mathcal{L}_{\text{bkg}} + \lambda_{\text{consist}} \mathcal{L}_{\text{consist}} + \lambda_{\text{smooth}} \mathcal{L}_{\text{smooth}} + 0.1 \mathcal{L}_{\text{text\_orth}} + \omega(e) \cdot \mathcal{L}_{\text{rupa}}$$

- $\mathcal{L}_{\text{mil}}$: Top-K MIL Loss với **Label Smoothing** ($0.02$) để triệt tiêu hiện tượng gradient bão hòa / loss về 0 gây overfit.
- $\mathcal{L}_{\text{clasm}}$: Cross-entropy đa nhãn với text prompts (`loss2_weight = 1.5`).
- $\mathcal{L}_{\text{consist}}$: Buộc detector score phải tương đồng với reconstruction error ($\lambda_{\text{consistency}} = 1.0$).
- $\mathcal{L}_{\text{bkg}}$: Ép prototype chỉ được phép phân loại vào lớp Normal ($\lambda_{\text{bkg}} = 0.5$).
- $\omega(e) = \min(1.0, e / \text{curriculum\_epochs})$: Curriculum learning ramp-up (3 epoch đầu) để làm ấm backbone trước khi siết chặt prototype.
- $\mathcal{L}_{\text{rupa}} = \lambda_{\text{gather}} \mathcal{L}_{\text{gather}} + \lambda_{\text{orth}} \mathcal{L}_{\text{orth}} + \lambda_{\text{res}} \mathcal{L}_{\text{residual}} + 0.5 \mathcal{L}_{\text{event}} + 0.2 \mathcal{L}_{\text{dnp\_normal}} + 0.01 \mathcal{L}_{\text{gate}}$.

---

## 4. Trạng thái & Kết quả Thực nghiệm gần nhất

### 4.1. UCF-Crime
- **Kỷ lục mới nhất:**
  - **Frame-level ROC-AUC:** **$86.76\%$** (tăng mạnh từ mức baseline $85.54\%$).
  - **Frame-level Average Precision (AP):** **$35.55\%$** (bứt phá ngoạn mục từ mức $28.41\%$).
  - **Semantics AUC:** Phục hồi từ $41.73\%$ lên **$86.35\%$**.
- **Cấu hình tối ưu:**
  - `topk-ratio = 16`
  - `temp = 2.0`
  - `smooth-sigma = 1.0`
  - `loss2_weight = 1.5`
  - `lambda-consistency = 1.0`
  - `lambda-bkg = 0.5`
  - `label-smoothing = 0.02`
  - `lr = 1e-4`
- **Mục tiêu tiếp theo:** Tinh chỉnh để chạm ngưỡng **$89.0\% - 90.0\%$ AUC**.

### 4.2. XD-Violence
- **Diễn biến vòng chạy trước:**
  - Tại **Epoch 2**: Mô hình bứt phá đạt **$\text{AP} = 78.09\%$ | $\text{AUC} = 93.08\%$** (đã lưu checkpoint `model_xd_best.pth`).
  - Tại Epoch 4-6: Do `label_smoothing = 0.0` và `lr = 2e-4`, MIL loss bị tụt về $0.0000$ $\rightarrow$ Bão hòa nhãn $\rightarrow$ Sinh ra False Positives khiến AP giảm về $62\%$.
- **Giải pháp triệt để đã cập nhật (Commit `e9be173`):**
  - Đặt `batch-size = 96` (48 samples/GPU trên Dual T4).
  - Thêm `label-smoothing = 0.02` và hạ `lr = 1.2e-4`.
  - Đặt `temp = 2.0`, `smooth-sigma = 1.2`.
  - Tích hợp hệ thống **Multi-Stream Ensemble** trong `src/xd_test.py`:
    - `prob_ens_rupa`: Ensemble giữa Routing và Semantics RUPA.
    - `prob_weighted`: Tỷ lệ 60% Routing + 40% RUPA Semantics.
    - `prob_geom`: Trung bình nhân hình học để dập nhiễu False Positives.
- **Mục tiêu hiện tại:** Chạy huấn luyện lại 12 epochs với BS 96 để đạt **$86.0\% - 87.5\%$ AP**.

---

## 5. Hướng dẫn Chạy Huấn luyện trên Kaggle Dual T4 (T4x2)

### 5.1. Khởi chạy XD-Violence (Batch Size 96 - Mục tiêu 86-87% AP)
```python
import subprocess, sys
from pathlib import Path

REPO_DIR = Path("/kaggle/working/RUPA-DSA")
ARTIFACTS = Path("/kaggle/working/rupa_artifacts")
xd_train_csv = ARTIFACTS / "xd_train.csv"
xd_test_csv = ARTIFACTS / "xd_test.csv"

# 1. Cập nhật mã nguồn mới nhất từ GitHub
print("🔄 Đang pull mã nguồn MA-RUPA mới nhất...", flush=True)
subprocess.run(["git", "pull", "origin", "main"], cwd=REPO_DIR, check=True)

# 2. Đường dẫn checkpoint
xd_model_sota_path = ARTIFACTS / "model_xd_sota_87.pth"
xd_checkpoint_sota_path = ARTIFACTS / "checkpoint_xd_sota.pth"

cmd_train_xd = [
    sys.executable, "-u", "src/xd_train.py",
    "--train-list", str(xd_train_csv),
    "--test-list", str(xd_test_csv),
    "--gt-path", "list/gt.npy",
    "--gt-segment-path", "list/gt_segment.npy",
    "--gt-label-path", "list/gt_label.npy",
    "--batch-size", "96",
    "--lr", "1.2e-4",
    "--max-epoch", "12",
    "--curriculum-epochs", "3.0",
    "--topk-ratio", "16",
    "--label-smoothing", "0.02",
    "--temp", "2.0",
    "--lambda-aux", "0.8",
    "--lambda-smooth", "0.3",
    "--loss2_weight", "1.5",
    "--lambda-consistency", "1.0",
    "--lambda-bkg", "0.5",
    "--seed", "234",
    "--model-path", str(xd_model_sota_path),
    "--checkpoint-path", str(xd_checkpoint_sota_path),
    "--smooth-eval", "True",
    "--smooth-sigma", "1.2"
]

print("🚀 Bắt đầu huấn luyện XD-Violence...", flush=True)
process = subprocess.Popen(cmd_train_xd, cwd=REPO_DIR, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
for line in iter(process.stdout.readline, ''):
    print(line, end='', flush=True)
process.stdout.close()
process.wait()
```

### 5.2. Khởi chạy UCF-Crime (Mục tiêu 89-90% AUC)
```python
import subprocess, sys
from pathlib import Path

REPO_DIR = Path("/kaggle/working/RUPA-DSA")
ARTIFACTS = Path("/kaggle/working/rupa_artifacts")
ucf_train_csv = ARTIFACTS / "ucf_train.csv"
ucf_test_csv = ARTIFACTS / "ucf_test.csv"

# 1. Cập nhật mã nguồn
subprocess.run(["git", "pull", "origin", "main"], cwd=REPO_DIR, check=True)

# 2. Checkpoint
ucf_model_path = ARTIFACTS / "model_ucf_sota_90.pth"
ucf_checkpoint_path = ARTIFACTS / "checkpoint_ucf_sota.pth"

cmd_train_ucf = [
    sys.executable, "-u", "src/ucf_train.py",
    "--train-list", str(ucf_train_csv),
    "--test-list", str(ucf_test_csv),
    "--gt-path", "list/gt_ucf.npy",
    "--gt-segment-path", "list/gt_segment_ucf.npy",
    "--gt-label-path", "list/gt_label_ucf.npy",
    "--batch-size", "64",
    "--lr", "1e-4",
    "--max-epoch", "15",
    "--curriculum-epochs", "3.0",
    "--topk-ratio", "16",
    "--label-smoothing", "0.02",
    "--temp", "2.0",
    "--lambda-aux", "0.2",
    "--lambda-smooth", "0.4",
    "--loss2_weight", "1.5",
    "--lambda-consistency", "1.0",
    "--lambda-bkg", "0.5",
    "--seed", "234",
    "--model-path", str(ucf_model_path),
    "--checkpoint-path", str(ucf_checkpoint_path),
    "--smooth-eval", "True",
    "--smooth-sigma", "1.0"
]

process = subprocess.Popen(cmd_train_ucf, cwd=REPO_DIR, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
for line in iter(process.stdout.readline, ''):
    print(line, end='', flush=True)
process.stdout.close()
process.wait()
```

---

## 6. Cấu trúc Thư mục và Các File Quan trọng

```
RUPA-DSA_final/
├── src/
│   ├── models/
│   │   ├── marupa.py          # Kiến trúc mạng chính (MDTBackbone, ModalityAdaptiveBranch, GradientSafeRUPA, QuadSafeGate)
│   │   ├── visual_net.py       # Transformer temporal encoder (Branch A)
│   │   ├── dnp.py              # Dynamic Normal Prototype memory & gather loss
│   │   └── text_encoder.py     # CLIP text encoder adapter
│   ├── ucf_train.py            # Huấn luyện End-to-End cho UCF-Crime
│   ├── ucf_test.py             # Đánh giá đa luồng Frame-level ROC-AUC & AP cho UCF-Crime
│   ├── ucf_option.py           # Siêu tham số mặc định cho UCF-Crime
│   ├── xd_train.py             # Huấn luyện End-to-End cho XD-Violence (Batch 96, Audio-Visual)
│   ├── xd_test.py              # Đánh giá đa luồng Frame-level AP & AUC cho XD-Violence
│   ├── xd_option.py            # Siêu tham số mặc định cho XD-Violence
│   ├── dataset.py              # Dataset Loader hỗ trợ cả CSV và audio npy
│   └── utils.py                # Gaussian/Savitzky-Golay smoothing, metrics calculation
├── tests/
│   └── test_marupa.py          # Unit test bao phủ toàn bộ module (100% pass)
├── list/                       # Chứa ground truth (.npy) và danh sách clip
└── PROJECT_CONTEXT.md          # File này (Bản ghi ngữ cảnh chuyển giao dự án)
```

---

## 7. Các Lưu ý Kỹ thuật Quan trọng khi Tiếp tục Phát triển

1. **Vấn đề PyTorch DataParallel Reduction:**
   - Các hàm tính loss (`compute_topk_mil_loss`, `compute_clasm_loss`, `compute_bkg_loss`, `compute_consistency_loss`) phải luôn trả về tensor scalar 0-D (`.mean()`). Nếu trả về vector 1-D, `torch.nn.DataParallel` sẽ cố gắng gather theo chiều 0 gây ra UserWarning và làm sai lệch gradient weighting giữa 2 GPU T4.
2. **Đảm bảo tính tương thích Backward Compatibility:**
   - Khi chạy inference hoặc test, `test(eval_model, ...)` cần truyền `model.module if hasattr(model, 'module') else model` để tương thích cả chế độ Single-GPU, Multi-GPU DataParallel và CPU.
3. **Kiểm tra Unit Test trước khi Push:**
   - Chạy lệnh: `.venv\Scripts\python.exe tests/test_marupa.py` để đảm bảo không có lỗi phân rã chiều (dimension mismatch), NaN attention guard, hoặc gãy API.
