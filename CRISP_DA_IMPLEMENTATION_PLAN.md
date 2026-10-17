# Kế hoạch triển khai CRISP-DA (GTA5 → Cityscapes)

**Cập nhật triển khai: 15/10/2026.** Mục 15 chốt các hợp đồng code, cấu hình thử nghiệm và cổng kiểm tra trước mỗi mốc. Quyết định mới trong mục này được phân biệt với công thức đề xuất trong PDF. Đây là kế hoạch triển khai; chưa có baseline nào được train hoặc kiểm chứng trong workspace.

**Cập nhật mã nguồn 17/10/2026:** đã triển khai pipeline B0 local, CLI, notebook launcher và kiểm thử bằng dữ liệu giả. Đã kiểm tra MiT-B5 ImageNet trên GPU local ở đầu vào nhỏ; chưa chạy benchmark GTA5 → Cityscapes thật. Xem [README_B0.md](README_B0.md) để chạy và [B0_VALIDATION.md](B0_VALIDATION.md) để phân biệt kiểm thử implementation với kết quả baseline nghiên cứu.

## 0. Mục tiêu và thứ tự ưu tiên

Tài liệu này chuyển hai bản thiết kế CRISP-DA thành các phần việc có thể code, chạy và kiểm chứng. **Ưu tiên số 1 là baseline đơn giản, tái lập được**, sau đó mới thêm các đóng góp nghiên cứu. CRISP-DA trong hai PDF là *đề xuất phương pháp*, chưa phải kết quả thực nghiệm đã được xác nhận; mọi cải thiện phải được đo bằng cùng giao thức dữ liệu và đánh giá.

| Mốc | Model | Mục đích | Điều kiện để đi tiếp |
|---|---|---|---|
| B0 | SegFormer MiT-B5, chỉ học trên GTA5 có nhãn | Xác nhận dữ liệu, nhãn, train/eval/checkpoint | Train chạy ổn định; có mIoU Cityscapes val và checkpoint khôi phục được |
| B1 | B0 + EMA teacher + pseudo-label confidence threshold + weak/strong augmentation | **Baseline UDA chính** để so sánh CRISP-DA | mIoU/per-class IoU, coverage và loss được log; chạy lại từ checkpoint cho kết quả nhất quán |
| C1 | B1 + source multi-prototype bank | Có semantic reference và phân phối prototype | Bank hữu hạn, class/mode có mẫu, posterior hợp lệ |
| C2 | + các evidence, temporal memory, trust estimator (MTE/METE) | Đo reliability thay vì chỉ max-softmax | MTE tốt hơn confidence-only trên bài toán correctness của source-corruption validation |
| C3 | + Four-State Policy (FSPD) | ACCEPT/REVISE/SOFT/ABSTAIN | Masks đúng, correction/coverage được đo; không làm giảm chất lượng do bug |
| C4 | + target-gated bank update | Cho bank thích nghi target có kiểm soát | Không xuất hiện prototype poisoning rõ rệt |
| C5 | + Boundary Head, rồi BPCC | Hoàn thiện bản đầy đủ | Có ablation riêng cho từng thành phần |

**Quy tắc thực hiện:** không bật nhiều module mới trong cùng một thí nghiệm. Giữ B0 và B1 làm mốc so sánh cố định. Nếu B1 chưa chạy tốt, chưa triển khai C1–C5 vào vòng train chính.

## 1. Căn cứ từ hai PDF và các quyết định triển khai

- `CRISP_DA_full.pdf`: §2 xác định closed-set UDASS; §4–§12 mô tả 7 thành phần, tensor/gradient; §13 nêu 8 loss; §14–§16 nêu thứ tự train và invariant; §22–§27 nêu ablation, evaluation và V1 tối giản.
- `CRISP-DA_final1.pdf`: slide 10–31 mô tả kiến trúc, bank, METE/FSPD/BPCC; slide 33–35 nêu 7 thành phần trong objective; slide 36–37 nêu curriculum.

Các điểm cần chốt trước khi code, nhằm tránh hai cách hiểu trong tài liệu:

| Vấn đề | Quan sát trong PDF | Quyết định cho plan |
|---|---|---|
| Baseline | Bản dài §22 yêu cầu EMA self-training + global confidence; slide tập trung CRISP-DA đầy đủ | Làm B0 source-only **và** B1 EMA confidence threshold. B1 là baseline UDA chính. |
| Target boundary loss | Bản dài §13 có `L_Tbnd`; objective ở slide 35 không có | C5 mặc định dùng source boundary loss. `L_Tbnd` là thí nghiệm mở rộng riêng, chỉ bật sau khi kiểm tra mask và boundary pseudo-GT. |
| Conservative adaptation | Bản dài §14.3 cho ACCEPT target tham gia loss; slide 36 ghi Student chỉ tính trên source | Dùng giai đoạn *probe* chỉ source để kiểm tra routing, sau đó mới bật hard ACCEPT loss bảo thủ. Ghi rõ hai substage trong config/log. |
| Tên estimator | Bản dài dùng MTE; slide dùng METE | Một module `TrustEstimator`, cùng chức năng. |
| Kích thước trust input | Bản dài §9.1: 5 evidence + class embedding 16 = 21; §19 ghi 25–64–32–1 | Lấy **21→64→32→1** cho bản đầy đủ. Khi bỏ boundary ở V1: 4 + 16 = **20**. `25` cần được xem là lỗi/biến thể chưa đặc tả, không hard-code. |
| REVISE và conflict | Slide 25–29 dùng các margin có định nghĩa khác nhau; bản dài §10 nêu rule tổng quát | Chuẩn hóa hai biến: `top2_margin = p_proto[c1]-p_proto[c2]` và `teacher_gap = p_proto[c1]-p_proto[c_teacher]`. Viết state priority xác định, kiểm thử riêng. |
| `L_proto` | Bản dài muốn cập nhật feature Student; Teacher và bank là non-gradient | Tính bằng **Student strong-view feature** trên hard accepted/revised pixels; prototype parameters và target/masks được detach. |

Các công thức loss, threshold và lịch train trong PDF là điểm khởi đầu nghiên cứu; không coi giá trị hyperparameter nào là đã tối ưu.

## 2. Bài toán, dữ liệu và chống rò rỉ nhãn

- **Source:** GTA5 RGB + semantic label, remap về 19 Cityscapes train IDs (`0..18`), ignore index `255`.
- **Target train:** Cityscapes `leftImg8bit/train`, **không đọc nhãn train trong optimizer, pseudo-label generation, trust calibration hay bank update**.
- **Target validation:** Cityscapes `leftImg8bit/val` và `gtFine/val/*_gtFine_labelTrainIds.png` (hoặc chuyển từ `labelIds` bằng mapping được khóa). Chỉ dùng cho đánh giá/so sánh checkpoint; không đưa vào gradient hay pseudo-label training.
- **Target test:** `gtFine/test` là dummy/ignore annotations, không dùng để tính mIoU. Chỉ chạy inference nếu sau này cần submit chính thức.
- **Setting:** closed-set, 19 classes. Evaluation chính: Cityscapes val mIoU; kèm per-class IoU. Cố định danh sách ảnh và seed cho mọi ablation.

### 2.1 Kiểm tra dữ liệu trước khi train

1. Tạo manifest `{sample_id, image_path, label_path?, split, height, width}`. `sample_id` phải ổn định để temporal memory tra cứu đúng ảnh.
2. Kiểm tra cặp image/label GTA5, Cityscapes val; không bỏ qua file lỗi một cách im lặng.
3. Kiểm tra giá trị nhãn sau remap chỉ thuộc `{0..18, 255}`; in histogram pixel theo class và visualize ít nhất một batch RGB–mask–overlay.
4. Xác nhận preprocessing ảnh và mask: resize ảnh bilinear, mask nearest-neighbor; crop/flip áp dụng đồng bộ. Không chuẩn hóa màu trên label.
5. Ghi nhận số ảnh từng split, checksum/phiên bản manifest và mapping GTA5→Cityscapes. Vì hai PDF không cung cấp bảng mapping GTA5 chi tiết, **phải xác minh mapping từ bộ annotation đang dùng trước khi train**.

### 2.2 Local code → Molab GPU → cloud data

- Code phát triển ở local, version bằng Git; notebook Molab chỉ làm launcher/quan sát, gọi các Python module trong repository. Dùng cùng commit hash cho run và checkpoint.
- Dataset Cityscapes/GTA5 lưu bên ngoài máy cá nhân (ví dụ private Kaggle dataset như đã trao đổi). GPU Molab vẫn phải đọc byte dữ liệu; tải/chia shard vào ổ tạm của phiên khi cần. Kiểm tra disk trống **trước** khi tải toàn bộ ~81 GB; chỉ giải nén những phần train/val cần thiết và tránh giữ đồng thời ZIP + dữ liệu giải nén nếu ổ tạm không đủ.
- Đường dẫn dữ liệu, output và cache lấy từ config/env, không hard-code đường dẫn Windows hoặc Molab. Ví dụ `DATA_ROOT`, `RUN_DIR`, `CACHE_DIR`.
- Checkpoint, config, log và metrics phải đồng bộ ra lưu trữ bền vững ngoài Molab sau mỗi mốc; không dựa vào ổ tạm của phiên để resume. Không commit dataset, token hoặc checkpoint lớn vào Git.

## 3. Cấu trúc project đề xuất

```text
crisp-da/
├── configs/
│   ├── b0_source_only.yaml
│   ├── b1_ema_selftrain.yaml
│   └── c1_bank.yaml ... c5_full.yaml
├── notebooks/
│   └── molab_train.py            # marimo launcher; không chứa core logic
├── src/crisp_da/
│   ├── data/{gta5,cityscapes,mapping,transforms,manifest}.py
│   ├── models/{segmentor,ema,boundary,trust}.py
│   ├── memory/{prototype_bank,temporal}.py
│   ├── policy/{four_state,bpcc}.py
│   ├── losses/{segmentation,consistency,prototype,boundary}.py
│   ├── engine/{train,evaluate,checkpoint,logging}.py
│   └── utils/{seed,distributed,geometry}.py
├── scripts/{inspect_data,train,evaluate,export_predictions}.py
└── tests/                       # kiểm tra logic có rủi ro: mapping, warp, EMA, policy
```

Mỗi module trả tensor và metadata rõ ràng; mọi config lưu cùng checkpoint. Không phụ thuộc vào state của notebook để tiếp tục train.

## 4. Mốc B0 — source-only segmentation baseline (làm đầu tiên)

### 4.1 Phạm vi

- SegFormer-style MiT-B5 encoder + decode/segmentation head, output 19 logits. Chọn một implementation và một checkpoint pretrained cụ thể, ghi tên/phiên bản/checksum; không đổi backbone giữa B0/B1/C1–C5.
- Forward source `x_s → logits_s`; loss duy nhất `L_sup = CE(logits_s, y_s, ignore_index=255)`.
- AdamW, scheduler, AMP (ưu tiên BF16 nếu runtime hỗ trợ), gradient clipping nếu cần. Batch/crop được quyết định bằng benchmark VRAM thực tế trên Molab; bắt đầu bằng crop vừa đủ để pipeline chạy, rồi cố định thông số trước ablation. Các LR từ PDF (backbone khoảng `6e-5`, head khoảng `6e-4`, weight decay `0.01`) chỉ là **điểm khởi đầu**.
- Evaluation dùng giao thức cố định ở §15.3. Báo cáo chính từ checkpoint tại số optimizer steps đã chốt; lưu `last` để resume. Có thể lưu thêm `best_target_val` phục vụ phát triển, nhưng phải ghi rõ nếu dùng Cityscapes val GT để chọn checkpoint; không gọi đó là lựa chọn model hoàn toàn không có nhãn target.

### 4.2 Trình tự triển khai

1. Viết `mapping.py`, `gta5.py`, `cityscapes.py`, transforms, manifest. Chạy inspect dataset.
2. Viết model wrapper trả `logits` và, để dùng về sau, `dense_feature` ở H/4×W/4 (khoảng 256 channels). B0 chưa dùng feature trong loss. Nếu thư viện SegFormer trả feature khác, projector/fusion cần là module riêng và được ghi trong config.
3. Viết `train.py` source-only, `evaluate.py`, checkpoint `model/optimizer/scheduler/scaler/step/config/RNG state`.
4. Chạy smoke test vài batch: forward, backward, loss hữu hạn, ignore pixels không góp gradient, train trên tập rất nhỏ để xác nhận loss giảm.
5. Chạy B0 đầy đủ, tạo bảng mIoU/per-class IoU, tốc độ, VRAM, train loss curve, qualitative predictions.

**Definition of done B0:** lệnh train và evaluate chạy từ đầu tới cuối; một checkpoint restore cho cùng predictions/metric trong tolerance; nhãn và overlay đúng; có baseline mIoU Cityscapes val được lưu thành artifact.

## 5. Mốc B1 — EMA self-training baseline (baseline UDA chính)

### 5.1 Thuật toán tối thiểu

1. Khởi tạo Teacher bằng bản sao Student sau B0 warm-up; Teacher `eval()` và `requires_grad=False`.
2. Mỗi iteration lấy source batch có nhãn và target train batch không nhãn. Teacher forward một **weak target view** dưới `no_grad`.
3. Tạo `p_T = softmax(logits_T)`, `y_T = argmax(p_T)`, `q = max(p_T)`. Pixel có `q ≥ tau_global` được nhận, còn lại ignore.
4. Student forward source và **strong target view**. Lúc đầu giữ geometry weak/strong **giống nhau**, chỉ strong appearance augmentation (color jitter/blur); như vậy tránh lỗi warp khi thiết lập baseline. Khi thêm geometric augmentation, bắt buộc chuyển pseudo-label/mask sang strong coordinates.
5. `L_B1 = L_sup + lambda_h * masked_CE(student_target, y_T, q≥tau_global)`. Target labels/masks detach. Nếu không pixel hợp lệ, target loss = 0.
6. Sau `optimizer.step()`, cập nhật Teacher EMA (khởi đầu `alpha=0.999`, theo PDF). Không dùng gradient cho Teacher.

Chọn `tau_global` và `lambda_h` là config được khóa cho baseline, chạy sensitivity nhỏ được ghi lại. Giữ source loss trong suốt adaptation. Log `accepted_fraction`, class histogram của accepted pixels, confidence distribution, source loss, target loss, mIoU và thời gian/VRAM.

### 5.2 Kiểm chứng B1

- Một pixel dưới ngưỡng không tạo target CE; pixel trên ngưỡng tạo CE đúng class.
- Teacher thay đổi chỉ sau EMA step; Teacher không có gradient.
- Pseudo-label đúng tọa độ target strong view; thử ảnh có flip/crop nếu bật transform hình học.
- Target `gtFine/train` không bao giờ được mở trong training process.
- Có bảng B0 vs B1 trên cùng Cityscapes val protocol; lưu checkpoint và lệnh tái lập. **B1 không bắt buộc phải hơn B0 trước khi xem xét nguyên nhân, nhưng phải là một baseline tin cậy và được debug nếu sụt mạnh.**

## 6. Mốc C1 — Class-Conditional Multi-Prototype Bank

- Từ source GT và `dense_feature` của checkpoint B0/B1, lấy feature tại H/4×W/4. Downsample mask bằng nearest-neighbor, bỏ ignore và tránh trộn class tại boundary khi cần.
- Mỗi class có `K=3` mode theo bản dài §8: `mu[c,k,256]`, diagonal `var[c,k,256]`, `pi[c,k]`. Bootstrap bằng sampled source features + K-means; tính mean/variance/occupancy. Các class có ít mẫu phải có fallback hợp lệ, không để NaN.
- Query target feature bằng diagonal Mahalanobis distance; trả `d_teacher` cho Teacher class và `p_proto[C]` trên tất cả class. Dùng clamp/floor cho variance và log-sum-exp để ổn định. Kiểm tra temperature/scaling của `exp(-D/T_d)` bằng histogram, vì D trên 256 chiều có thể làm compatibility gần 0 toàn bộ.
- **C1 chỉ query, chưa nhận target update.** Source statistics có thể tiếp tục update ngoài gradient sau khi query. Bank lưu/restore trong checkpoint.
- C1 là mốc kiểm tra module. Nếu bank chỉ được query/log và chưa ảnh hưởng loss hoặc pseudo-label, segmentation phải giống B1 với cùng seed/input/state; không xem C1 là một thí nghiệm chứng minh tăng mIoU.
- Kiểm tra trên source validation: feature gần `mu[c,k]` có distance thấp; `p_proto` cộng thành 1; top-class confusion và mode occupancy được log. Đừng coi bank posterior là xác suất calibrated nếu chưa đo.

## 7. Mốc C2 — Evidence, temporal memory và Trust Estimator

### 7.1 Evidence map tại cùng hệ tọa độ H/4×W/4

| Ký hiệu | Cách tạo | Kiểm tra |
|---|---|
| `q` | max-softmax Teacher weak view | Trong `[0,1]` |
| `a` | `1 - JS(p_T_w1, aligned(p_T_w2))`, normalize | Identity views cho `a≈1` |
| `d` | Compatibility của Teacher class từ bank | Feature gần mode cho score cao hơn feature xa |
| `s` | `1 - JS(p_T_current, M_previous[id])` | Lần đầu gặp ID: dùng neutral value/mask, không tự so với memory vừa update |
| `b` | Boundary probability từ Teacher boundary head | Chỉ thêm ở C5; C2 dùng 4 evidence |

- Hai weak views nên chung crop/resize/flip, khác nhẹ ở màu để phép so sánh ban đầu đơn giản. Với geometry khác nhau phải lưu transform metadata và align probability maps trước JS. Strong-view targets cũng phải align.
- Temporal memory keyed bằng **image ID ổn định**; lưu probability thấp độ phân giải trên CPU/remote hoặc representation nén, với EMA `gamma≈0.9` theo PDF. Không cấp phát full-resolution `N_images×19×H×W`; tính bộ nhớ trước khi bật. Query `M_previous`, dùng để tính `s`, rồi mới update memory cuối iteration. Memory là state trong checkpoint; nếu quá lớn, có snapshot riêng và kiểm tra resume.
- TrustEstimator V1: input `q,a,d,s` + class embedding 16, pointwise MLP/1×1 conv `20→64→32→1`, sigmoid. Bản đầy đủ thêm `b` thành 21 chiều. Predicted class embedding dựa trên Teacher argmax.

### 7.2 Calibration training không cần target GT

Tạo corrupted source views (blur/fog/noise/brightness/color shift), chạy Teacher để lấy 4 evidence; correctness GT là `g_i = 1[argmax(p_T_corrupted)_i == y_source_i]` ở valid source pixels. Train TrustEstimator bằng class-balanced BCE/balanced sampling. **Dùng cùng quy tắc tạo evidence ở train và inference**, đặc biệt `d`, `a` và temporal score; nếu source không có lịch sử temporal, cần định nghĩa neutral/masked `s` nhất quán và báo ablation (4 evidence so với 3 evidence). Không giả định source corruption mô phỏng đầy đủ GTA→Cityscapes shift.

Tách source train/source calibration validation theo ảnh; không dùng Cityscapes train GT. So sánh MTE với `q` bằng AUROC/AUPRC cho correctness/error detection, ECE, Brier score, reliability diagram. C2 hoàn thành khi inference target tạo reliability map hữu hạn `[0,1]` và calibration proxy được báo rõ. Target val GT chỉ được dùng cho phân tích offline có gắn nhãn *diagnostic*, không huấn luyện MTE.

## 8. Mốc C3 — Four-State Pseudo-label Decision (FSPD)

### 8.1 Contract

Input: `p_T [B,19,h,w]`, `p_proto`, `r [B,1,h,w]`, `teacher_class`, class thresholds `tau_low[c] < tau_high[c]`. Output: `hard_target`, `soft_target`, `state` và pixel weights; các masks ACCEPT/REVISE/SOFT/ABSTAIN tạo partition, tổng bằng 1 tại mọi valid pixel.

### 8.2 Rule V1 xác định, có thể ablate

1. Loại pixel invalid hoặc `r≤tau_low` vào `ABSTAIN`. Bank phải vượt cổng bootstrap trước khi bật FSPD; nếu bank thiếu class/mode làm posterior không hợp lệ, dừng adaptation và báo lỗi có ngữ cảnh, thay vì âm thầm khiến toàn bộ batch ABSTAIN.
2. `REVISE` chỉ khi `tau_low<r<tau_high`, `c1≠c_teacher`, `p_proto[c1]≥eta`, `top2_margin≥delta_rev`, `teacher_gap≥delta_gap`; nhãn hard = `c1`, weight `w_R<1`. V1 không thêm `tau_very_high` vì đã có upper bound `tau_high`.
3. Nếu chưa REVISE mà có `strong_conflict` theo §15.7, chuyển `ABSTAIN`; nếu không, `ACCEPT` khi `r≥tau_high`, nhãn hard = Teacher class, weight 1.
4. `SOFT` nếu còn `tau_low<r<tau_high`, bank hợp lệ và không có conflict cực mạnh; `p_soft = alpha*p_T + (1-alpha)*p_proto`, `alpha = clamp((r-tau_low)/(tau_high-tau_low),0,1)` theo slide 27. Nếu bank chưa đáng tin, không trộn bừa: để `ABSTAIN` trong V1.

Thứ tự code thực tế: loại invalid/low reliability → kiểm tra REVISE → ACCEPT → SOFT → ABSTAIN. Đây là một **quyết định triển khai cụ thể** để giải quyết xung đột giữa slide 25–29 và §10 bản dài; lưu thresholds trong config để sau đó làm sensitivity/ablation.

### 8.3 Loss và kiểm tra

- `L_hard`: masked CE cho ACCEPT + `w_R` cho REVISE; chia tổng pixel weights hợp lệ. Không có pixel hard thì trả scalar zero trên đúng device.
- `L_soft`: masked KL/soft CE trên SOFT, target distribution detach và normalize. Không có pixel soft thì zero.
- `L_cons`: weak Teacher → strong Student KL, chỉ trên pixel được phép, có reliability weight; không cho ABSTAIN đóng góp trong V1.
- Unit tests: 4 states đúng ở điểm biên threshold; `sum(masks)=1`; `sum_c p_soft=1`; top-2 bank bằng nhau không REVISE; `r` rất cao với conflict mạnh không sửa bừa; các target tensors align strong geometry; masks 0 không tạo gradient.
- Log theo class: state fractions, hard/soft coverage, mean reliability, correction attempts. Correction precision trên **target val offline** nếu muốn nghiên cứu; không dùng metric này để tạo nhãn target train.

## 9. Mốc C4 — Target-gated bank adaptation

- Trong một iteration, **query bank trước** khi state decision; sau Student optimizer và EMA Teacher mới update bank bằng detached features của batch đó. Không để target pixel đánh giá chính nó sau khi đã update bank.
- Giai đoạn đầu: source-only bank. Sau C3 ổn định: chỉ nhận target ACCEPT; dùng trọng số target thấp hơn source (`lambda_T<1`, slide 19 khuyên `<0.5` như điểm khởi đầu). REVISE chỉ xét ở thí nghiệm muộn với ngưỡng rất nghiêm và weight nhỏ; SOFT/ABSTAIN không cập nhật bank.
- Mỗi `(class,mode)` chỉ update khi đủ mẫu; variance floor, occupancy smoothing và EMA `m≈0.99`; không merge/split modes ở V1. Lưu `mu/var/pi/count/step` vào checkpoint.
- So sánh source-only bank với target-adapted bank bằng per-class mode occupancy, `d`, accepted target fraction, mIoU và correction precision. Nếu bank dịch sai class hoặc mode collapse, dừng target update và trở lại source anchor.

## 10. Mốc C5 — Boundary Head và BPCC (sau V1)

**Boundary Head:** nhận feature H/4, dự đoán boundary logit/probability. Tạo source boundary GT từ khác biệt nhãn lân cận hoặc morphological gradient trên *label IDs*, không lấy số học `dilate(label)-erode(label)` trực tiếp nếu làm mất ý nghĩa class; ignore pixels không tạo cạnh giả. Source loss = BCE + Dice. Thêm `b` vào TrustEstimator thành input 21 chiều, nên cần huấn luyện/calibrate lại MTE. Bản slide dừng tại source boundary loss; `L_Tbnd` từ PDF dài chỉ bật trong ablation riêng khi pseudo-boundary và Teacher boundary đồng thuận.

**BPCC:** theo dõi hard/soft coverage và reliability trung bình theo Teacher-predicted class. Bản đầu dùng class-wise quantile + safety floor từ §11.5, cập nhật theo window; controller feedback ở §11.4 là ablation riêng. `tau_low[c] = tau_high[c]-delta_tau` với clamp để luôn `<tau_high`. Class ít mẫu giữ threshold cũ, không ép quota nếu reliability thấp. Lưu toàn bộ threshold trajectory và so sánh với fixed thresholds.

## 11. Loss schedule và gradient routing

| Giai đoạn | Loss đang bật | State được update |
|---|---|---|
| B0 | `L_sup` | Student |
| B1 | `L_sup + lambda_h L_hard_confidence` | Student gradient; Teacher EMA |
| C1 bank bootstrap/probe | Như B1 | Bank source statistics ngoài gradient |
| C2 calibration | Như B1 + `lambda_r L_trust` trên corrupted source | TrustEstimator gradient riêng; target reliability dùng `detach` |
| C3 probe | Source loss + MTE calibration; routing chỉ log | Chưa đưa target four-state loss vào Student |
| C3 full V1 | `L_sup + lambda_h L_hard + lambda_s L_soft + lambda_c L_cons + lambda_p L_proto + lambda_r L_trust` | Student, TrustEstimator; Teacher EMA; bank theo C4 nếu bật |
| C5 | Thêm `lambda_sb L_Sbnd`; optional `lambda_tb L_Tbnd` ablation | Boundary Head gradient; BPCC state ngoài gradient |

Các `lambda` phải được ghi trong config; loss mở rộng bắt đầu từ 0 rồi bật dần, source segmentation loss luôn có hệ số 1. Với `L_proto`, Student feature giữ gradient, bank detached. Với `L_trust`, correctness target tính từ prediction detached và source GT; không cho pseudo-label routing gradient quay về MTE/Teacher/Bank. Các loss theo mask chia **tổng weight hợp lệ + epsilon**, không chia toàn bộ ảnh, và không có valid pixel phải ra zero chứ không NaN.

## 12. Thứ tự một iteration CRISP-DA đầy đủ

```text
source_batch, target_batch = next(...), next(...)
student(source) -> source logits/features/boundary -> L_sup, L_Sbnd
teacher(no_grad, target weak views) -> p_T, F_T, b_T
align weak views -> q, a
bank.query(F_T) -> d, p_proto                 # bank hiện tại, CHƯA update
temporal.read(image_id) -> s                   # memory của iteration trước
trust(q,a,d,s,b,class) -> r
policy(r,p_T,p_proto,thresholds) -> hard/soft targets + masks
align targets to strong view; student(strong target)
compute active target losses + source-corruption L_trust
optimizer.step(student, boundary, trust as configured)
EMA.update(teacher, student)
bank.update(detached source + gated target features)
temporal.update(image_id, detached p_T)
stats.update(); BPCC.update() ở cuối window/epoch
checkpoint/log theo cadence
```

## 13. Evaluation, ablation và tiêu chí hoàn thành

**Mỗi run:** lưu commit hash, config, seed, manifest hash, checkpoint nguồn/pretrained, GPU, crop/batch/AMP, số iteration, thời gian, peak VRAM và best/last checkpoint. Đánh giá bằng cùng Cityscapes val resize/sliding-window protocol; không đổi protocol giữa các hàng trong bảng.

| Nhóm | Metric tối thiểu |
|---|---|
| Segmentation | mIoU, per-class IoU, head/tail class IoU, qualitative overlays |
| Pseudo-label | hard/soft/abstain coverage theo class, confidence/reliability histogram, precision–coverage trên target val offline |
| Trust | AUROC/AUPRC, ECE, Brier trên source-corruption validation; target val offline diagnostic tách biệt |
| Revision | Số pixel REVISE, correction precision và hại/lợi so với Teacher label trên target val offline |
| Boundary | Boundary-IoU/trimap metric nếu bật C5 |
| Hệ thống | Wall time/iteration, peak VRAM, disk/RAM/cache, resume thành công |

**Bảng ablation:** triển khai module theo B0→B1→C1…C5, nhưng chạy thí nghiệm theo §15.9 để mỗi hàng có thay đổi thật sự trong supervision/loss. Bank query-only là kiểm tra module, không phải ablation cải thiện mIoU. Cố định số optimizer steps và báo riêng GPU-hours, vì thêm Teacher/evidence forwards làm tăng chi phí dù số steps bằng nhau. Chạy nhiều seed cho kết luận cuối khi tài nguyên cho phép; không kết luận phương pháp tốt hơn chỉ từ một run nhiễu.

## 14. Ưu tiên công việc ngay tiếp theo

1. Khóa manifest, GTA5→Cityscapes mapping và Cityscapes val label source; làm inspect script.
2. Chọn implementation SegFormer MiT-B5 và pretrained checkpoint; viết model wrapper/dataloader/evaluation.
3. Chạy **B0 source-only** trước, lấy mIoU đầu tiên và checkpoint restore được.
4. Bật **B1 EMA confidence baseline** với một weak view và strong appearance view; đo lại mIoU/coverage.
5. Chỉ khi B0/B1 ổn định mới bắt đầu source bank C1 và trust/policy C2–C3.

**Sản phẩm đầu tiên cần có:** một lệnh train B0, một lệnh evaluate, `b0_source_only.yaml`, manifest/mapping đã kiểm tra và bảng Cityscapes val mIoU. Đây là nền để mọi phần của CRISP-DA sau này có đối chứng đáng tin.

## 15. Quyết định triển khai và các sửa đổi sau review

### 15.1 Kiến trúc và pretrained: chốt trước benchmark B0

- Dùng implementation MiT-B5 đã có pretrained weights; không tự viết lại attention/Mix-FFN ở B0. File `models/segmentor.py` chứa adapter cho encoder và các head của project.
- **Pretraining chính dùng ImageNet, không dùng segmentation checkpoint đã học với Cityscapes GT.** Ghi nguồn và training provenance của checkpoint; checkpoint không rõ nguồn chưa được dùng cho kết quả UDA chính.
- Giữ đúng kiến trúc custom của bản dài: bốn projector về 256 chiều, fusion 1024→256, segmentation head 3×3 256→256 rồi 1×1 256→19. Nạp pretrained cho encoder; projector/head khởi tạo mới. Không gọi head này là nguyên bản của mọi implementation SegFormer.
- Quyết định code ban đầu cho các Norm mới trong decoder/head: GroupNorm 32 groups + GELU, phù hợp batch nhỏ; giữ nguyên norm trong encoder pretrained. Đây là lựa chọn triển khai, phải dùng nhất quán từ B0 tới các ablation. Nếu đổi decoder/head hoặc Norm sau B0, cần chạy lại baseline tương ứng.
- Trả `logits_low [B,19,h,w]`, `logits [B,19,H,W]`, `dense_features [B,256,h,w]`. `dense_features` lấy từ fusion trước classifier/dropout, cùng định nghĩa cho Student và Teacher.
- Trước benchmark, điền `encoder_library`, `library_version`, `pretrained_uri`, `pretrained_hash`, `pretraining_dataset`, `load_report` vào manifest. Đây là cổng triển khai cần xác minh khi chọn dependency thực tế; plan chưa xác nhận một package/checkpoint cụ thể đã tải và load thành công.

### 15.2 Data contract và split cố định

- Dataset trả một sample không có batch dimension: `image float32[3,H,W]`, `label int64[H,W]` nếu có, `sample_id str`, `original_hw`, `transform_meta`. DataLoader mới tạo `[B,...]`.
- Config bắt buộc có `label_encoding: raw_ids | train_ids`. Chỉ remap khi `raw_ids`; kiểm tra nhãn đã là train IDs để tránh remap hai lần. Nhãn không nằm trong mapping phải thành `255` hoặc gây lỗi theo policy được ghi rõ.
- Bảng mapping phải được viết và xác minh bằng annotation thực tế, có tên class, raw ID, train ID; không suy từ màu PNG hiển thị. Cityscapes `labelTrainIds` nếu chưa có thì tạo từ `labelIds`, giữ nguyên file gốc.
- Trước B0, tách source thành `source_train` và `source_dev` theo ảnh/nhóm cảnh nếu có metadata; ghi toàn bộ sample IDs và hash trong manifest. Dùng cùng split cho mọi run. Calibration source ở C2 tiếp tục tách theo ảnh thành calibration train/dev, không lấy pixel từ cùng ảnh vào cả hai tập.
- B0 chỉ tạo GTA5 train/dev loader và Cityscapes val loader. Các target-train loader chỉ khởi tạo từ B1. Overlay phải kiểm tra cả class name, ignore regions và cặp ảnh–mask.

### 15.3 Evaluation và model selection

- Giao thức chính B0: mỗi ảnh Cityscapes val được resize tới một `eval_hw` cố định đã ghi trong config, forward toàn ảnh, rồi bilinear-upsample **logits** về kích thước GT gốc (`align_corners=False`) trước argmax. Không resize GT để làm khớp prediction. Không test-time flip, multi-scale hay sliding window trong run chính đầu tiên.
- Chọn `eval_hw` qua kiểm tra VRAM, rồi khóa cho B0/B1/CRISP. Nếu cần sliding window để đạt độ phân giải cao hơn, coi đó là một eval protocol mới và chạy lại toàn bộ checkpoint so sánh.
- Tích lũy confusion matrix int64 trên toàn tập; bỏ GT `255`; tính IoU từng class từ tổng confusion matrix, rồi lấy trung bình 19 class theo quy ước đã ghi. Không lấy trung bình mIoU từng ảnh/batch. Class có union bằng 0 phải được xử lý nhất quán và báo rõ.
- Run chính dùng checkpoint ở `max_steps` đã định trước; tuning dựa trên source-dev/source-corruption-dev và diagnostics không nhãn target. Đánh giá target val theo cadence cố định, không sửa config giữa run theo val score.
- Run phát triển có thể chọn `best_target_val`, nhưng phải báo đây là lựa chọn bằng nhãn validation target và tách khỏi kết quả fixed-step. Không dùng cùng bảng kết quả mà bỏ qua khác biệt model-selection protocol.

### 15.4 B0: cấu hình thử đầu và thứ tự code

Các giá trị dưới đây là **mặc định chạy thử do plan đề xuất**, chưa phải cấu hình đã đạt chất lượng tốt trên Molab. Sau preflight, khóa lại trước run chính.

| Trường | Quyết định ban đầu |
|---|---|
| Train | Một GPU; crop 512×512; source batch 2; accumulation 1; seed 42 |
| Augmentation | Giữ tỉ lệ ảnh khi resize theo scale range được khai báo; random crop/flip đồng bộ với mask; normalization theo pretrained encoder |
| Optimizer | AdamW; backbone LR 6e-5; projector/head LR 6e-4; weight decay 0.01; parameter groups liệt kê rõ, không bỏ sót/đếm hai lần parameter |
| Scheduler | Linear warm-up rồi polynomial decay; bắt buộc điền `warmup_steps`, `max_steps`, `power`, `min_lr` trước run chính |
| Precision | BF16 nếu runtime hỗ trợ và forward/backward hữu hạn; nếu dùng FP16 phải có GradScaler; CPU smoke dùng FP32 |
| Smoke | 20 optimizer steps để kiểm tra train/eval/checkpoint; overfit 2–4 source images để kiểm tra learning signal |
| Run chính | `max_steps` chọn sau khi đo seconds/step và quota phiên; không coi smoke run là baseline đã hoàn thành |
| Logs | Loss/LR/step time mỗi 20 steps; eval và checkpoint mỗi 1000 optimizer steps như điểm bắt đầu; thêm checkpoint ở cuối run |

Một `step` luôn là **optimizer step**. Với gradient accumulation, scheduler và EMA chỉ chạy sau optimizer step thành công; nếu FP16 scaler bỏ step do overflow thì cũng không chạy EMA/scheduler. Thời gian nguồn, eval và calibration phải được tính vào dự toán GPU-hours. Nếu preflight buộc đổi crop/batch, ghi config cuối và giữ nguyên cho phép so sánh sau đó.

Thứ tự code B0: `config/seed` → `mapping/manifest` → `dataset/transforms` → `segmentor` → `CE/optimizer/scheduler` → `trainer` → `evaluator` → `checkpoint/logging` → `CLI/Molab launcher`. Viết kiểm tra có ý nghĩa cho mapping, geometry, metric confusion matrix và resume. Chỉ chuyển B1 sau một run B0 đủ budget đã chốt, có kết quả evaluate và phục hồi được.

### 15.5 Tensor resolution, loss và EMA contract

- Source CE và B1 hard target CE chạy ở H×W. Từ CRISP V1, policy/evidence/target hard-soft-consistency/prototype loss chạy ở h×w = H/4×W/4; không trộn mask h×w với logits H×W. Khi thay loss resolution so với B1, thêm control B1 ở h×w để đo ảnh hưởng của thay đổi resolution.
- Lấy `p_T_low = softmax(logits_low)` để tạo Teacher class/confidence/evidence ở h×w. Không âm thầm downsample argmax từ full resolution rồi giả định nó giống argmax của low-resolution distribution.
- Tất cả probability targets sau warp phải renormalize theo class trên vùng valid; hard labels/state masks dùng nearest. Strong appearance augmentation giữ geometry ở B1; CutMix/ClassMix để sau khi có tests chuyển đồng bộ image/label/probability/state/weight và donor metadata.
- Cố định hướng KL là `KL(target || student)`. Empty masked loss dùng zero có nối graph, ví dụ `student_logits.sum()*0`, để backward an toàn; source batch toàn ignore phải báo và bỏ optimizer step, không tính mean CE gây NaN.
- Teacher deep-copy toàn Student (cộng boundary head khi bật), freeze và `eval`. EMA float parameters; copy các non-floating buffers và chốt policy float buffers theo model thực tế. Kiểm tra tên/shape, không dựa riêng vào `zip(parameters())` khi hai kiến trúc có thể khác nhau.

### 15.6 Bank và MTE: kiểm soát feature space và calibration

- Bootstrap bank bằng **Teacher.eval source features** và source GT; target query cũng dùng Teacher.eval features. Khi update online, source anchor lấy từ cùng Teacher feature extractor, bổ sung source no-grad forward nếu cần. Không gộp Student train-mode source features với Teacher eval-mode target features rồi mặc định hai không gian thống kê giống nhau. `L_proto` vẫn kéo Student strong features về bank tham chiếu.
- Bản tham chiếu giữ score của PDF `logsumexp(log pi - D/2)` và gọi là **prototype class score đã softmax**, chưa tuyên bố Gaussian posterior đã calibration. Thêm biến thể Gaussian đầy đủ `logsumexp(log pi - (D + sum(log(var)))/2)` khi muốn kiểm tra ảnh hưởng covariance normalization; không đổi công thức giữa run mà giữ nguyên tên thí nghiệm.
- Khóa feature preprocessing (raw hay normalized) từ bootstrap tới query/update/loss. Ban đầu dùng raw fused feature để theo đặc tả; tính distance/variance/log-sum-exp bằng FP32, dùng pixel chunks để tránh tạo tensor khổng lồ `[B,h,w,C,K,D]`. `T_d`, variance floor và `lambda_p` chọn qua source-dev và scale diagnostics; log `L_proto` cùng gradient norm trước khi bật vào objective.
- Không dùng covariance 0 cho mode thiếu dữ liệu. C1 phải có ít nhất một mode hợp lệ cho mỗi class trước FSPD; thiếu class thì tiếp tục thu source features. Với K=3 nhưng mode rỗng, đánh dấu invalid và loại khỏi logsumexp, không coi centroid giả là reference.
- Weighted BCE/balanced sampling có thể làm thay đổi xác suất correctness ước lượng. Sau khi train MTE, calibrate trên held-out source-dev với tần suất correct/incorrect tự nhiên bằng loss không reweight, rồi đo ECE/Brier. Reliability sau calibration mới dùng làm threshold input. Không đồng nhất AUROC tốt với xác suất đã calibrated.
- C2 ban đầu freeze Teacher/bank để kiểm tra MTE. Khi adaptation làm feature/prediction distribution đổi, duy trì source-corruption calibration theo cadence được ghi trong config; đánh giá lại trên source-dev, không dùng target pseudo-label correctness làm GT của MTE.

### 15.7 Temporal memory và FSPD: định nghĩa các trường hợp biên

- Temporal là thành phần của V1 khi bật `s`, không chờ tới Full. State gồm image ID, canonical grid shape (ban đầu H_original/16×W_original/16), probability FP16, spatial valid/visit mask và update count. Chỉ cập nhật vùng crop đã quan sát; vùng chưa từng thấy không được hiểu là stable.
- Khi chưa có memory, đặt `s=0.5` như fallback **được khai báo**, truyền `memory_valid` trong metadata để log/mask phép tính temporal. Khi train MTE, cũng áp cùng fallback cho missing history; không chỉ train toàn `s=1` rồi chạy target với history thật. Việc thêm `memory_valid` thành input neural là ablation kiến trúc riêng; canonical input vẫn 20 chiều V1, 21 chiều khi thêm boundary.
- Source calibration cần prediction history qua các lần thăm ảnh bằng cùng Teacher/version schedule và cùng quy tắc crop-alignment như target; correctness GT so với source label đã align. Nếu chưa thực hiện được, chạy biến thể không temporal `q,a,d` (19 input channels với embedding) và ghi đúng tên; không gọi là V1 đủ bốn evidence.
- FSPD chọn thresholds theo **raw Teacher class** trong iteration hiện tại. Định nghĩa `top2_margin`, `teacher_gap` như §1; `strong_conflict = (proto_top != teacher_class) AND (teacher_gap >= delta_conflict)`.
- Rule cụ thể: invalid/`r<=tau_low` → ABSTAIN; sau đó REVISE nếu đủ mọi điều kiện §8.2; sau đó `strong_conflict` chưa được revision → ABSTAIN; sau đó `r>=tau_high` → ACCEPT; còn lại → SOFT. Bỏ optional `tau_very_high` ở V1 vì REVISE đã bị giới hạn `r<tau_high`; chỉ thêm nếu mở rộng vùng REVISE và có ablation.
- Các so sánh `>`/`>=` phải giữ đúng config/spec; test `r==tau_low`, `r==tau_high`, conflict vừa đúng ngưỡng, bank top-2 tie. Mỗi valid pixel có đúng một state; pixel ngoài support không góp loss/statistics.

### 15.8 Boundary, BPCC và full objective

- Source boundary: tạo binary edge bằng sự khác class giữa valid neighbors, dilate binary edge theo band đã khai báo; không dùng chênh lệch số raw class IDs làm boundary target liên tục.
- Target boundary loss vẫn OFF trong Full mặc định theo slide; là extension thứ tám của objective từ bản dài. Muốn bật phải có cả reliable positive edges và reliable negative interiors, đồng thuận với Teacher boundary head, và loại vùng lân cận ignore/ABSTAIN. Không tính BCE chỉ trên pixel edge dương.
- Viết hai config rõ tên: `full_source_boundary.yaml` (7 thành phần objective) và `full_target_boundary_extension.yaml` (thêm `L_Tbnd`). Không dùng số lượng loss để định nghĩa V1 đã hoàn thành.
- BPCC counts dùng cùng tập pixel được nhóm theo Teacher class: `coverage[c] = count((teacher_class==c) AND hard_state)/count(teacher_class==c)`. Log thêm histogram **refined label class** để thấy REVISE chuyển supervision sang class nào; không trộn hai cách đếm trong cùng tỉ số.
- Quantile update theo window có `min_samples_per_class`, safety floor và `delta_tau`; class thiếu mẫu giữ threshold cũ. Quantile điều khiển candidate coverage; FSPD conflict/revision filters khiến hard coverage thực tế có thể khác quota. Log cả hai, không tuyên bố quantile bảo đảm exact hard coverage sau routing.

### 15.9 Thí nghiệm và phục hồi phiên Molab

| Run | Thay đổi tác động tới training | Control cần giữ |
|---|---|---|
| E0 | B0 source-only | Architecture, pretrained provenance, source split, eval protocol |
| E1 | B1 confidence Keep/Ignore | Cùng E0 architecture; ghi riêng warm-up và adaptation steps |
| E1-low | B1 target CE ở h×w | Control cho thay đổi resolution của CRISP |
| E2 | Calibrated MTE confidence-only thay confidence trong Keep/Ignore | Chưa bật FSPD/`L_proto`; cùng calibration split và target loss |
| E3 | Thêm `a`, rồi `d`, rồi `s` qua các run riêng | Recalibrate mỗi estimator; giữ policy Keep/Ignore và budget |
| E4 | FSPD thay Keep/Ignore với cùng MTE/bank | Chỉ thêm hard correction + soft branch; giữ target bank update OFF |
| E5 | Thêm `L_proto`, rồi `L_cons` qua hai run riêng | Giữ thresholds/features/routing |
| E6 | Source-only bank → ACCEPT target update | Cùng bank scoring và loss |
| E7 | Thêm source boundary head/evidence | Train/calibrate lại MTE, ghi chi phí warm-up bổ sung |
| E8 | Thêm BPCC | Cùng E7, thay fixed thresholds |
| E9 | Optional target boundary extension | Cùng E8, kiểm tra positive/negative boundary masks |

- B1 và các run adaptation bắt đầu từ cùng checkpoint B0 đã chọn theo protocol §15.3. Báo tổng source warm-up steps + adaptation steps, không chỉ báo phần adaptation. So sánh B0 train tiếp trên source cùng tổng steps nếu muốn tách gain adaptation khỏi gain do train lâu hơn.
- Pin environment sau smoke test trên GPU, ghi Python/PyTorch/CUDA/dependency versions và GPU model. Kiểm tra forward/backward thật trên GPU đã cấp trước khi chạy dài; không suy khả năng tương thích từ tên GPU.
- Checkpoint gồm model, optimizer, scheduler, AMP state khi có, global optimizer step, phase/counters, RNG Python/NumPy/CPU/CUDA, sampler/cursor policy, EMA, MTE/calibration state, bank, BPCC và temporal snapshot/version. Khôi phục RNG mà không khôi phục sampling/augmentation policy chưa đủ để hứa tái lập từng batch.
- B0/B1 ưu tiên resume tại epoch boundary hoặc sampler cursor xác định. Nếu chỉ hỗ trợ restart đầu epoch, ghi rõ có lặp data và không tuyên bố resume bitwise. Test so sánh một số bước liên tiếp trước/sau resume với cùng input stream và tolerance thích hợp.
- Lưu checkpoint tạm rồi atomic rename; upload ra durable storage với checksum/version, xác nhận đọc lại được. Định kỳ checkpoint trong run, không phụ thuộc vào hook khi session bị ngắt đột ngột. Dataset download/cache tách khỏi training module, có preflight disk/RAM và manifest validation.

**Cổng bắt đầu code B0:** có thể code các module ngay theo contract này. **Cổng bắt đầu benchmark B0:** mapping/manifest được kiểm tra, package/checkpoint đã load đúng, config không còn placeholder, resource preflight và metric/resume tests đã đạt. Các quyết định bank/MTE/FSPD không chặn tiến độ B0.
