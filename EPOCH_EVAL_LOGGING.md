# Lưu evaluation theo epoch trong B0

`train.eval_every_epoch: true` bật đánh giá **source_dev và Cityscapes target_val
sau mỗi lượt duyệt hết source_train**. Cityscapes vẫn chỉ dùng để đánh giá,
không tham gia gradient. Model-selection chính vẫn là fixed optimizer steps.

```yaml
train:
  eval_every_epoch: true
  eval_every: 0
```

`eval_every: 0` tắt lịch theo steps. Đặt số dương nếu muốn thêm evaluation theo
steps. Khi epoch end và lịch steps/final trùng nhau, mỗi split chỉ đánh giá một
lần, với nhiều trigger. Cuối max_steps luôn đánh giá, kể cả dừng giữa epoch.

## File được ghi ngay trong lúc train

```text
RUN_DIR/
  eval_history.jsonl             lịch sử chuẩn, mỗi record một split/lần eval
  eval_history.csv               bảng tiện để vẽ biểu đồ
  eval/step_<n>/<split>/metrics.json
  eval/step_<n>/<split>/per_class_iou.csv
```

CSV có `epoch`, `completed_epochs`, `source_position`, `step`, `split`, `trigger`,
`loss_ce`, `miou`, `pixel_accuracy`, `samples`, `valid_pixels`, `classes_with_union`
và 19 cột `iou_road`…`iou_bicycle`. IoU/accuracy là tỉ lệ 0..1; nhân 100 khi vẽ %.
Class có union=0 lưu null trong JSON, ô trống trong CSV. Nếu toàn GT ignore thì
loss CE/accuracy/mIoU null thay vì NaN hoặc 0 giả tạo.

CE validation được tính từ logits resize về GT gốc, tổng CE / tổng valid pixels
toàn split. Đây là validation loss; train CE hiện vẫn log theo optimizer steps
trong `metrics*.jsonl`, không phải trung bình train loss của cả epoch.

`epoch=1.0` là đã duyệt hết source train một lần. `epoch=1.4` là đang giữa lượt
thứ hai; có thể xuất hiện nếu bật eval theo steps hoặc khi kết thúc max_steps.
Muốn chỉ lấy điểm cuối epoch, lọc `trigger` chứa `epoch_end`.

Một đoạn 500 steps để backup không phải một epoch. Lịch sử được nối xuyên các
đoạn resume và tự loại record trùng ở cùng step/source cursor/split. Khi resume
checkpoint cũ trong cùng thư mục còn log tương lai, những record vượt quá
step/cursor của checkpoint bị loại trước khi tiếp tục.

Với accumulation, nhóm microbatch cuối epoch có thể ngắn hơn config để không
đọc sang epoch tiếp theo trước khi đánh giá; CE vẫn normalize theo valid pixels.
Source sampler/cursor đã có trong checkpoint nên không reset epoch khi resume.
Epoch được tính theo ảnh đã đọc, không theo số optimizer updates thành công;
all-ignore/FP16 overflow vẫn không tăng optimizer step.

Xem `data.json` để biết source train count và `nominal_optimizer_steps_per_epoch`.
Không có skip thì số updates/epoch xấp xỉ:

```text
ceil(ceil(source_train_count / batch_size) / grad_accumulation)
```

Ví dụ khoảng 23.700 ảnh source train, batch 2, accumulation 1 thì cần khoảng
11.850 steps để hết một epoch. Run 1.000 steps chỉ có điểm cuối run ở epoch
phân số, chưa có điểm cuối epoch đầy đủ. Muốn đồ thị nhiều epoch phải chọn
budget tương ứng; không suy chất lượng baseline chỉ từ số epoch đã chạy.

## Cell xem bảng trên MoLab, không cần cài thêm package

```python
import csv as _csv

with (RUN_DIR / "eval_history.csv").open(newline="", encoding="utf-8") as _stream:
    _rows = list(_csv.DictReader(_stream))
for _row in _rows:
    print({key: _row[key] for key in ("epoch", "step", "split", "trigger", "loss_ce", "miou")})
```

## Cell vẽ mIoU và validation CE theo epoch

Cell này dùng matplotlib; nếu môi trường chưa có, cài `matplotlib` bằng package
manager MoLab hoặc `subprocess.run([sys.executable, "-m", "pip", "install",
"matplotlib"], check=True)`. Không đưa matplotlib thành dependency của trainer.

```python
import csv as _csv
import matplotlib.pyplot as _plt

with (RUN_DIR / "eval_history.csv").open(newline="", encoding="utf-8") as _stream:
    _rows = list(_csv.DictReader(_stream))
_figure, _axes = _plt.subplots(1, 2, figsize=(12, 4))
for _split in ("source_dev", "target_val"):
    _selected = sorted(
        (_row for _row in _rows
         if _row["split"] == _split and "epoch_end" in _row["trigger"].split("+")),
        key=lambda row: float(row["epoch"]),
    )
    for _axis, _metric, _scale in ((_axes[0], "miou", 100), (_axes[1], "loss_ce", 1)):
        _points = [(float(row["epoch"]), _scale * float(row[_metric]))
                   for row in _selected if row[_metric] != ""]
        if _points:
            _axis.plot([p[0] for p in _points], [p[1] for p in _points], marker="o", label=_split)
for _axis, _ylabel in zip(_axes, ("mIoU (%)", "Validation CE")):
    _axis.set_xlabel("Completed source epoch")
    _axis.set_ylabel(_ylabel)
    _axis.grid(alpha=0.3)
    if _axis.lines:
        _axis.legend()
_figure.tight_layout()
_figure.savefig(RUN_DIR / "eval_by_epoch.png", dpi=160)
_figure
```

Nếu chưa có epoch hoàn chỉnh, đường vẽ trống là đúng. Có thể bỏ điều kiện
`epoch_end` để xem thêm các điểm theo step/final ở epoch phân số.

## Đưa code mới sang MoLab

Ở Git Bash local, đưa các thay đổi này lên GitHub:

```bash
cd /f/CRISP_DA
git add src/crisp_da configs tests README_B0.md B0_VALIDATION.md EPOCH_EVAL_LOGGING.md GTA_SIZE_MISMATCH_DIAGNOSTIC.md MOLAB_B0_GTA_01_03_RUNBOOK.md MOLAB_B0_GTA_01_10_RUNBOOK.md
git commit -m "Log B0 evaluation metrics at source epoch boundaries"
git push origin main
```

Commit/push code local trước. Runbook mới resolve `GIT_REF="main"` thành SHA
đầy đủ rồi pin cho run mới; khi resume dùng đúng SHA đã ghi trong snapshot.
Code cũ ở commit `02d2d5a` chưa có feature này. Không chỉ sửa YAML rồi chạy trên
checkout cũ. Bật feature này cho run mới và giữ nguyên config khi resume.
Checkpoint cũ thiếu flag được hiểu là `eval_every_epoch=false`; không tự bật
flag khi resume, đặc biệt nếu dùng accumulation. Không có lịch sử epoch cũ thì
không thể tái tạo đúng metric từng epoch chỉ từ checkpoint cuối.

Snapshot trong runbook đóng gói toàn RUN_DIR nên bao gồm cả hai file history.
Để ảnh plot cuối cũng có trên Drive, chạy cell verified backup sau khi vẽ.
