# CRISP-DA B0 — hướng dẫn chạy mã nguồn

GTA RGB half-size: CLI `make-manifests --gta-upsample-image-id 15188` cho phép
riêng RGB nhỏ đúng 1/2 nhãn ở ID đã kiểm tra được upsample bilinear khi đọc.
Mask/PNG gốc giữ nguyên; manifest ghi policy, dimensions và SHA256 để inspect
và loader áp dụng nhất quán. Mismatch khác vẫn lỗi. Xem
[chẩn đoán và cách áp dụng](GTA_SIZE_MISMATCH_DIAGNOSTIC.md), rồi kiểm tra overlay.

Evaluation theo epoch: bật `train.eval_every_epoch: true`; `eval_every: 0` chỉ
đánh giá cuối mỗi source epoch và cuối run. Số dương thêm cadence theo steps.
`eval_history.csv`/`eval_history.jsonl` lưu mIoU, validation CE, pixel accuracy,
19 IoU, epoch/step và split xuyên các đoạn resume. Xem
[hướng dẫn và cell vẽ biểu đồ](EPOCH_EVAL_LOGGING.md). Runbook yêu cầu commit/push
code mới để MoLab dùng trainer có feature này; checkout cũ không tự có history.

Mapping update (2026-10-17): raw ID `34` observed in GTA masks is mapped to
`255` (ignored), since it is outside the benchmark's 19 raw IDs. This is
GTA-specific and applies to manifest validation, inspection and training.
Cityscapes validation and rejection of other unexpected IDs remain strict.
The manifest report records `gta_extra_raw_ignore_ids: [34]`.
This follows the exclusion policy in the
[AdaptSegNet GTA loader](https://github.com/wasidennis/AdaptSegNet/blob/master/dataset/gta5_dataset.py);
it does not assert a semantic class name for ID 34. Inspect real overlays
before marking the dataset mapping as verified.

Mã nguồn thực hiện **source-only GTA5 → Cityscapes**, theo mục 4 và 15 của
[kế hoạch](CRISP_DA_IMPLEMENTATION_PLAN.md). B0 có CLI train/evaluate, dữ liệu manifest,
mapping, kiểm tra ảnh–nhãn, resume và notebook launcher. Kết quả trên dữ liệu thật
chỉ được coi là baseline khi đã chạy benchmark đầy đủ; kiểm thử giả không thay thế việc đó.

## 1. Những gì được triển khai

- Encoder: Hugging Face Transformers **4.57.6**, `nvidia/mit-b5`, revision
  `40357155205b036cf11b61f132d53d2f8861f170`. Model card ghi pretraining ImageNet-1k;
  không lấy segmentation checkpoint Cityscapes. [Nguồn weights](https://huggingface.co/nvidia/mit-b5).
- Decoder của project: bốn projector 256 → fusion 1024→256 → head 3×3 và classifier 19 lớp;
  các block mới dùng GroupNorm(32) + GELU. `dense_features` nằm trước classifier.
- Outputs: `logits_low`, `logits` ở kích thước đầu vào và `dense_features` 256 chiều.
- Source CE với ignore 255; AdamW hai nhóm LR; warmup + poly; accumulation tính theo
  tổng valid pixels; FP32/BF16/FP16 với GradScaler. All-ignore bỏ update; overflow FP16
  không tăng optimizer step và scheduler.
- Source train/dev rời nhau; B0 chỉ đọc source và Cityscapes val. Không tạo target-train loader.
- Eval full-image, một scale; bilinear logits về GT gốc, confusion int64 toàn tập.
  Class union=0 có IoU `null`, loại khỏi trung bình và báo số class hiện diện.
- Checkpoint đầy đủ, sampler cursor, RNG và augmentation theo sample/epoch. CPU test
  so sánh resume với luồng liên tục; không hứa bitwise giữa GPU/phần cứng khác nhau.

## 2. Cài đặt local

Ở PowerShell, tại `F:\CRISP_DA` (nếu chưa có môi trường):

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev,notebook]"
```

Cần PyTorch >=2.6 tương thích GPU/CUDA của máy; trên MoLab bản PyTorch đã kiểm tra
thành công có thể được giữ lại. Không dùng `--upgrade` toàn bộ môi trường một cách mặc định.
Trong lần triển khai hiện tại, `.venv` local dùng các package torch/numpy/Pillow có sẵn
qua `--system-site-packages`; dependency bổ sung được cài trong `.venv`.

Kiểm tra CLI:

```powershell
.venv\Scripts\python.exe -m crisp_da.cli --help
```

## 3. Dữ liệu thật: từ ZIP đến manifest

Luồng xác thực Drive/tải ZIP nằm trong [hướng dẫn local → MoLab](CRISP_DA_LOCAL_TO_MOLAB_GUIDE.md).
CLI B0 nhận **ZIP đã tải về** hoặc dữ liệu đã giải nén; không tự có quyền truy cập Drive.

Xem cấu trúc ZIP bằng Python trước khi chọn nơi giải nén. Lệnh sau bảo toàn đường dẫn
thành viên ZIP, giới hạn số byte giải nén, đọc CRC cho các member được chọn và yêu cầu
destination mới:

```powershell
.venv\Scripts\python.exe -m crisp_da.cli extract-zip cache/01_images.zip --destination tmp/extracted/01_images --max-uncompressed-gib 20
```

Có thể lặp `--include "pattern"` để chọn thành viên. Giới hạn 20 GiB ở đây là **ngưỡng do
bạn đặt**, không chứng minh MoLab có đủ quota. Kết quả nằm ở staging riêng; sau khi
xem thư mục bên trong, sắp xếp ảnh/nhãn theo layout dưới, kiểm tra trùng tên trước khi gộp.
Chưa cung cấp tiện ích tự động gộp các ZIP GTA vì cấu trúc thực bên trong chưa được xác minh.

```text
DATA_ROOT/
  gta/images/**/*.png
  gta/labels/**/*.png
  cityscapes/leftImg8bit/val/<city>/*_leftImg8bit.png
  cityscapes/gtFine/val/<city>/*_gtFine_labelIds.png
```

GTA ghép theo tên file không có `.png`. ID trùng trong cùng nhóm hoặc cặp thiếu sẽ gây lỗi.
Với Cityscapes train IDs, dùng hậu tố `_gtFine_labelTrainIds.png` và chọn `train_ids`.
Mask phải là ảnh chỉ số 2D (P/L hoặc integer), không phải PNG màu RGB. Mapping mặc định
là raw Cityscapes IDs 7,8,11,...,33 → 0..18; phải kiểm tra GTA annotation thật dùng đúng
quy ước này. Các raw void IDs chính thức thành 255; raw ID lạ gây lỗi ở cấu hình mặc định.
[Mapping chính thức](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py).

Ví dụ với dataset giải nén ở `data/`:

```powershell
.venv\Scripts\python.exe scripts/prepare_data.py --data-root data --gta-encoding raw_ids --city-encoding raw_ids
.venv\Scripts\python.exe scripts/inspect_data.py --data-root data --manifest data/manifests/source_train.jsonl --output runs/inspect-source
.venv\Scripts\python.exe scripts/inspect_data.py --data-root data --manifest data/manifests/target_val.jsonl --output runs/inspect-val
```

`prepare_data` scan toàn bộ ảnh/nhãn, tạo source train/dev theo ảnh với seed 42, target val,
report histogram và hash manifest. Chưa có metadata cảnh nên split là image-level random;
không tuyên bố đã tách theo nhóm cảnh. Xem các PNG RGB–mask–overlay và `inspection.json`.
Nếu chỉ muốn kiểm tra nhanh, thêm `--limit 8`; báo cáo sẽ ghi rõ đây là scan một phần.

Manifest chứa `label_encoding`; config phải trùng encoding của từng manifest. Thay mapping,
split hoặc manifest cần một run mới, không resume trên dữ liệu đã thay đổi. Hash manifest
không phát hiện mọi thay đổi byte ảnh; giữ dataset bất biến/version hóa và kiểm tra hash
archive gốc trong quá trình chuẩn bị dữ liệu.

## 4. Chạy thử hoàn toàn offline bằng dữ liệu giả

Luồng này dùng một encoder SegFormer nhỏ khởi tạo ngẫu nhiên, **không phải MiT-B5**,
chỉ để kiểm tra implementation và không tải pretrained:

```powershell
.venv\Scripts\python.exe -m crisp_da.cli synthetic-data
.venv\Scripts\python.exe scripts/train.py --config configs/b0_tiny_test.yaml
.venv\Scripts\python.exe scripts/evaluate.py --checkpoint tmp/synthetic/run/last.pt --output tmp/synthetic/eval --device cpu
```

Nếu thư mục synthetic/run đã có checkpoint, dùng `--resume` hoặc chọn `RUN_DIR` mới;
không ghi đè âm thầm. Dữ liệu giả có 6 ảnh source và 2 ảnh val kích thước khác nhau.

## 5. Chạy MiT-B5 pretrained và dữ liệu thật trên MoLab

Sau khi repository và package đã có trong phiên MoLab, chạy trong terminal Linux:

```bash
python -m pip install -e .
python -m crisp_da.cli lock-encoder --cache-dir /marimo/crisp-work/cache --output configs/mit_b5_imagenet.lock.json
export DATA_ROOT=/marimo/crisp-work/data
export CACHE_DIR=/marimo/crisp-work/cache
export RUN_DIR=/marimo/crisp-work/runs/b0-smoke-001
python scripts/train.py --config configs/b0_smoke.yaml
python scripts/evaluate.py --checkpoint "$RUN_DIR/last.pt" --data-root "$DATA_ROOT" --output "$RUN_DIR/evaluate-fixed-step"
```

`lock-encoder` tải config, processor và weights (~328 MB), dùng revision cố định và ghi
SHA256. File lock là metadata nhỏ, có thể version hóa trong Git. `build_model` nạp encoder
strict, bỏ classifier ImageNet và ghi báo cáo load. Không âm thầm bỏ qua missing keys.
Đối chiếu mean/std normalization với processor của pretrained checkpoint.

`configs/b0_smoke.yaml` chạy 20 optimizer steps với B5, crop 512×512 và batch 2;
đánh giá sau 10/20 bước. Có thể dùng manifest subset nhưng phải ghi phạm vi dữ liệu.
Nếu GPU lỗi kernel hoặc hết VRAM, điều chỉnh sau đo thực tế; không suy khả năng chạy
toàn model chỉ từ kiểm tra CUDA nhỏ.

Nếu dùng cell notebook thay terminal, dùng `subprocess.run` với danh sách arguments,
`sys.executable`, `cwd` của repository và `env` chứa các paths. Không dán lệnh `export`
trực tiếp vào cell Python. Có launcher ở [notebooks/molab_train.py](notebooks/molab_train.py).
Notebook này không thực hiện clone, OAuth hoặc đồng bộ Drive thay bạn; các bước đó
phải hoàn thành theo hướng dẫn. Cài thêm `.[notebook]` nếu chạy marimo local.

## 6. Chốt benchmark B0

Sao chép `configs/b0_source_only.yaml` thành cấu hình run riêng trong `configs/`, rồi điền:

1. `train.max_steps` và `train.warmup_steps` phù hợp với timing đã đo.
2. `run.budget_note`: GPU, seconds/step, chi phí eval và budget lựa chọn.
3. `model.pretrained_lock: configs/mit_b5_imagenet.lock.json`.
4. `data.mapping_verified: true` **sau khi** kiểm tra annotation/overlay.
5. Paths manifests, encoding, crop/scale và `eval_hw` đã khóa.

Config benchmark ban đầu cố ý để budget chưa xác định và mapping chưa xác nhận;
CLI sẽ từ chối chạy nếu các cổng này chưa đạt. Smoke config chạy được trước các cổng
benchmark; kết quả smoke không được gắn nhãn baseline nghiên cứu đã hoàn thành.

Giữ `save_best_target_val: false` cho protocol fixed-step chính. Nếu bật, checkpoint
`best_target_val.pt` là lựa chọn dựa trên nhãn target val và phải báo riêng. Dữ liệu source dev
được đánh giá cùng cadence để hỗ trợ debug/tuning source. Không thay config trong run theo
target val score. B0 không thực hiện calibration hoặc dùng target train GT.

## 7. Checkpoint, artifacts và resume

Mỗi run có:

```text
config.json, environment.json, data.json, encoder_provenance.json
metrics.jsonl, loss_curve.svg, summary.json, last.pt
eval/step_<n>/source_dev/metrics.json
eval/step_<n>/target_val/metrics.json
eval/.../per_class_iou.csv
eval/.../*_trainIds.png, *_color.png, *_rgb_gt_prediction.png
```

IoU lưu theo tỷ lệ 0..1; nhân 100 nếu báo phần trăm. Ảnh triptych dùng RGB đầu vào eval
đã đưa về kích thước GT để xem cùng prediction. `seconds_per_step` gồm đọc batch,
forward/backward/update, không gồm eval; VRAM là peak allocated của PyTorch, không
phải toàn bộ VRAM process/driver. Logs resume nằm trong file riêng có step bắt đầu.

```bash
python scripts/train.py --config configs/b0_smoke.yaml --stop-after 7
python scripts/train.py --config configs/b0_smoke.yaml --resume /path/to/last.pt
```

`--stop-after` là optimizer step tuyệt đối; không đổi `max_steps` để mô phỏng pause,
vì như vậy thay LR schedule. Resume kiểm tra contract config và hash các manifest;
cho phép đổi paths, device và workers. Muốn đổi budget hoặc thuật toán thì tạo run mới.
Độ chính xác số học có thể khác khi đổi device/precision.

Checkpoint được ghi tạm và atomic rename ở ổ local. Đồng bộ `last.pt` đã ghi xong cùng
metadata về Drive, xác minh readback/hash rồi mới xem là bản khôi phục bền vững.
CLI hiện **chưa có upload Drive tự động trong lúc train**: trước run dài cần bổ sung
uploader theo cơ chế Drive đã xác thực hoặc runner đồng bộ định kỳ. Không dựa vào cell
upload chỉ chạy sau train. Việc này không ảnh hưởng kiểm thử B0 local nhưng là cổng
vận hành cho các run MoLab dài có nguy cơ mất phiên.

Chỉ load full training checkpoint do bạn tạo/tin cậy; full state có Python/NumPy RNG.
Evaluate từ checkpoint không tải lại pretrained và chỉ cần manifest/dữ liệu của split đã chọn.

## 8. Kiểm thử và giới hạn hiện tại

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Tests kiểm tra mapping/void IDs, crop/flip đồng bộ, GT eval nguyên kích thước, ghép cặp,
source split, không đọc target train GT, sampler chống prefetch lệch cursor, ZIP traversal,
tensor contract, ignore gradient, confusion tổng, optimizer groups, LR và resume.
Có kiểm tra overfit fixed source batch với model nhỏ; không thay thế overfit B5 trên ảnh GTA thật.

`configs/b0_b5_preflight.yaml` là cấu hình kiểm tra B5 thật ở 64×64 với dữ liệu giả.
Tạo fixture và tải encoder trước khi dùng; chạy bằng `scripts/train.py` như các config khác.
Nó yêu cầu GPU và chỉ dùng file pretrained đã cache (`local_files_only: true`).

Trạng thái xác minh của lần triển khai được ghi ở `B0_VALIDATION.md`.
Không có kết quả mIoU GTA→Cityscapes thật trước khi tải/kiểm tra dataset và chạy trên MoLab.
Quota ổ đĩa của phiên vẫn chưa được xác định từ báo cáo overlay 8 EiB.
