# Trạng thái kiểm chứng B0

**GTA RGB alignment (19/10/2026):** trường hợp người dùng quan sát ở `15188`
là RGB 957×526, mask palette 1914×1052. CLI opt-in theo ID chỉ phóng RGB đúng 2×
bilinear khi đọc, giữ nguyên mask/PNG. Policy, sizes và hashes nằm trong manifest;
loader và inspect dùng chung đường đọc. **23 tests đạt**, gồm default strict,
unknown ID/tỷ lệ khác bị từ chối, palette bytes giữ nguyên và phát hiện input
bị thay đổi sau khi tạo manifest. Vẫn cần xem overlay thực tế trên MoLab.

**Cập nhật 19/10/2026 — epoch evaluation:** đã bổ sung `eval_every_epoch`,
validation CE theo valid pixels, pixel accuracy và `eval_history.csv/jsonl` với
mIoU/19 IoU, epoch và optimizer step. Bộ kiểm thử đạt **20 tests**; sau thay đổi
metadata và bổ sung kiểm tra runtime epoch-only, hai test liên quan được chạy
lại và đạt. Đã kiểm tra epoch boundary với accumulation, nối lịch sử và model
state tương đương khi resume, gộp trigger trùng và loại log tương lai khi rollback.
Đây là kiểm chứng CPU tiny/local, chưa phải benchmark B5 MoLab trên GTA đầy đủ.
Xem [hướng dẫn](EPOCH_EVAL_LOGGING.md).

Ngày: **17/10/2026**. Phạm vi: implementation local; chưa có kết quả benchmark GTA5 → Cityscapes.

## Môi trường dùng để kiểm chứng local

- Windows, Python 3.14.6; `.venv` có sử dụng system site packages.
- PyTorch 2.13.0+cu130; Transformers 4.57.6.
- GPU NVIDIA GeForce RTX 4060 Laptop GPU, khoảng 8 GiB VRAM.
- Đây là máy local, **khác** runtime MoLab người dùng đã kiểm tra: PyTorch 2.11.0+cu130,
  RTX PRO 6000 Blackwell ~95 GiB và RAM container 160 GiB.

## Kiểm tra implementation

Các kiểm tra được thực thi bằng pytest trong workspace; xem test cases ở `tests/`:

- Mapping 19 lớp, void/ignore, raw IDs so với train IDs và từ chối mask RGB.
- Crop/flip ảnh–mask đồng bộ, GT giữ kích thước gốc khi evaluate.
- Pairing, source train/dev rời nhau, manifest duplicate/path traversal.
- B0 không truy cập nhãn Cityscapes train, kể cả khi đặt file GT train hỏng trong fixture.
- Cursor sampler không bị đẩy bởi prefetch.
- Giới hạn giải nén, traversal ZIP và xử lý thư mục đích đã tồn tại.
- Tensor contract và gradient zero tại pixel ignore; all-ignore không tạo optimizer update.
- Confusion matrix tích lũy toàn tập, chính sách class absent.
- Bilinear logits trước argmax tại kích thước GT gốc.
- Optimizer parameter groups, LR warmup/poly.
- Resume giữa epoch khớp model weights và optimizer state của luồng CPU liên tục.
- Resume từ chối manifest thay đổi; model nhỏ học được fixed source batch.

**Lượt xác minh cuối: 15 tests PASS trong 33,42 giây.** Lệnh:

```text
python -m pytest -q --basetemp=tmp/pytest-b0-release
```

Không suy khả năng bitwise resume giữa phần cứng khác nhau từ kiểm thử CPU cùng môi trường.

## Pretrained MiT-B5 thật

Đã tải weights ImageNet-1k từ `nvidia/mit-b5`, revision cố định
`40357155205b036cf11b61f132d53d2f8861f170` và nạp strict encoder thành công.

SHA256 weights:

```text
a389c0a604458fa205446fd08a2c01b74e6591a5da3e77de668c6ca5cbf75356
```

Hash config/processor và weights được lưu ở
`configs/mit_b5_imagenet.lock.json`. Decoder/head mới khởi tạo theo mục 15.1 của plan;
classifier ImageNet bị loại, không nạp segmentation checkpoint Cityscapes.

Đã chạy `configs/b0_b5_preflight.yaml` trên GPU local với ảnh/nhãn **giả**, crop 64×64,
batch 1, 2 optimizer steps, precision auto (BF16 trên GPU local):

- Forward, loss, backward và AdamW hoạt động; không có skipped update.
- Loss source trong hai bước: khoảng 3,0103 → 1,3781. Hai batch khác nhau nên đây không
  phải phép đo overfit hoặc bằng chứng hội tụ trên dữ liệu thật.
- Peak PyTorch allocated VRAM khoảng 1,55 GiB tại cấu hình nhỏ này.
- Evaluate source dev/target val giả, ghi metrics và prediction PNG, lưu full checkpoint.
- Evaluate độc lập từ checkpoint B5 cho metrics/confusion matrix giống lần evaluate
  trong train trên cùng dữ liệu giả; không cần tải lại weights pretrained.

Không dùng VRAM/timing ở 64×64 để dự đoán batch/crop 512×512 trên MoLab.
Không báo các IoU giả như kết quả Cityscapes thật.

## Artifact được tạo local

Các file sau nằm dưới `tmp/`, bị loại khỏi Git và chỉ là bằng chứng kiểm thử:

```text
tmp/synthetic/data/                 # fixture nhân tạo
tmp/synthetic/run/                  # CLI tiny model, 4 steps
tmp/synthetic/standalone-eval/      # evaluate checkpoint độc lập
tmp/synthetic/inspection/          # histogram và overlay qua CLI inspect-data
tmp/b5-preflight/                   # B5 ImageNet, 2 steps ở đầu vào nhỏ
```

Mỗi run có config, môi trường, hash manifest, provenance, log, checkpoint và các kết quả eval.
Không cam kết các artifact tạm sẽ còn nếu thư mục `tmp/` được người dùng dọn.

## Cổng còn lại trước benchmark thật

- Xác thực Drive trong MoLab; tải và kiểm tra nội dung ZIP thật, không chỉ tên file.
- Kiểm tra GTA label encoding và overlay; tạo full manifests/source split cố định.
- Xác minh storage MoLab phù hợp; số overlay 8 EiB không xác định quota thực tế.
- Kiểm tra toàn bộ model/dependency trên runtime MoLab, crop/batch/eval_hw dự kiến.
- Chốt max_steps, warmup, budget và xác nhận mapping cho cấu hình benchmark.
- Thiết lập đồng bộ checkpoint định kỳ về Drive và thử readback/resume thực tế.
- Chạy B0 đầy đủ, lấy mIoU/per-class IoU Cityscapes val theo protocol cố định.

Chỉ sau các bước trên mới có thể đánh dấu Definition of Done của **baseline nghiên cứu B0**.
