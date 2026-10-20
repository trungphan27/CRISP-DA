# GTA image/label size mismatch: kiểm tra trước khi sửa dữ liệu

`ValueError: Image/label size mismatch: gta/15188` nghĩa là Pillow đọc được
hai file nhưng width/height khác nhau. `CalledProcessError` chỉ là thông báo
subprocess CLI đã thất bại. Đây không phải lỗi GPU hoặc mapping raw IDs.

Chưa có kích thước và đối chiếu bytes của hai file nên chưa thể kết luận file
nào sai, ảnh bị đổi sau giải nén hay sự lệch đã có trong ZIP. Không tự resize,
xóa hoặc bỏ qua cặp này. PNG raw ID cần giữ đúng hình học; chỉ đổi kích thước
không chứng minh hai nội dung tương ứng.

## Cell chẩn đoán 15188: đối chiếu PNG local với ZIP

Chạy sau Cell 9 và trước khi chạy lại Cell 10. Cell chỉ đọc file và ghi report
JSON nhỏ. `DATA` và `ARCHIVES` là hai biến public ở Cell 1 runbook. Các biến
`_...` còn lại đều nội bộ trong cell; không yêu cầu biến private của cell trước.
PNG có thể là hard link tới data subset cũ, nên chưa sửa file trực tiếp.

```python
from pathlib import Path as _Path
from PIL import Image as _Image
import hashlib as _hashlib
import io as _io
import json as _json
import zipfile as _zipfile

_sample_id = "15188"
_report = {"sample_id": _sample_id, "local": {}, "zip_members": {}}

def _sha_file(_path):
    _digest = _hashlib.sha256()
    with _path.open("rb") as _stream:
        for _block in iter(lambda: _stream.read(4 * 1024**2), b""):
            _digest.update(_block)
    return _digest.hexdigest()

for _kind in ("images", "labels"):
    _paths = list((DATA / "gta" / _kind).rglob(_sample_id + ".png"))
    if len(_paths) != 1:
        raise RuntimeError(f"Cần đúng một {_kind}/{_sample_id}.png, tìm thấy: {_paths}")
    _local = _paths[0]
    with _Image.open(_local) as _image:
        _image.load()
        _local_info = {
            "path": str(_local), "width": _image.width, "height": _image.height,
            "mode": _image.mode, "sha256": _sha_file(_local),
        }
    _report["local"][_kind] = _local_info
    _matches = []
    for _archive in sorted(ARCHIVES.glob(f"*_{_kind}.zip")):
        with _zipfile.ZipFile(_archive) as _zip:
            for _member in _zip.infolist():
                if (not _member.is_dir()
                        and _Path(_member.filename).name == _sample_id + ".png"):
                    # Đọc member đến EOF: zipfile cũng kiểm tra CRC.
                    _bytes = _zip.read(_member)
                    with _Image.open(_io.BytesIO(_bytes)) as _image:
                        _image.load()
                        _entry = {
                            "archive": str(_archive), "member": _member.filename,
                            "width": _image.width, "height": _image.height,
                            "mode": _image.mode,
                            "sha256": _hashlib.sha256(_bytes).hexdigest(),
                        }
                    _entry["local_matches_zip"] = _entry["sha256"] == _local_info["sha256"]
                    _matches.append(_entry)
    _report["zip_members"][_kind] = _matches

_report_path = DATA / f"size_mismatch_{_sample_id}.json"
_report_path.write_text(_json.dumps(_report, indent=2), encoding="utf-8")
print(_json.dumps(_report, indent=2))
print("Report:", _report_path)
```

Ý nghĩa kết quả:

- `local_matches_zip=false`: PNG local không còn giống archive đang kiểm tra.
  Xác minh đúng archive/version và đường dẫn trước khi khôi phục member sang
  một thư mục mới; không ghi đè file hard link đang dùng ở run khác.
- Cả hai `true` nhưng width/height trong ZIP khác nhau: sự lệch đã có trong
  cặp archive. Giải nén lại cùng ZIP sẽ không sửa được. Cần tìm cặp annotation
  đúng hoặc chốt ngoại lệ dataset có ghi rõ ID, lý do, số ảnh và hash manifest.
- Không tìm thấy member hoặc có hơn một member cùng ID: kiểm tra thiếu/trùng
  archive và version. Không tự chọn một bản theo thứ tự file.

## Cell tùy chọn: scan header toàn GTA để liệt kê tất cả cặp lệch

Cell này chỉ đọc header để tìm mismatch kích thước và lỗi mở PNG, nhanh hơn
full decode/scan nhãn của make-manifests. Nó không thay thế kiểm tra CRC,
decode đầy đủ hay kiểm tra label IDs. Report chưa tự quyết định loại ảnh nào.

```python
from PIL import Image as _Image
import json as _json

_index = {}
for _kind in ("images", "labels"):
    _index[_kind] = {}
    for _path in sorted((DATA / "gta" / _kind).rglob("*.png")):
        if _path.stem in _index[_kind]:
            raise RuntimeError(f"Duplicate {_kind} ID: {_path.stem}")
        _index[_kind][_path.stem] = _path
if _index["images"].keys() != _index["labels"].keys():
    raise RuntimeError("Thiếu ảnh/nhãn; cần kiểm tra ghép cặp trước")
_issues = []
for _number, _id in enumerate(sorted(_index["images"]), 1):
    _sizes = {}
    try:
        for _kind in ("images", "labels"):
            with _Image.open(_index[_kind][_id]) as _image:
                _sizes[_kind] = list(_image.size)  # [width, height]
        if _sizes["images"] != _sizes["labels"]:
            _issues.append({"sample_id": _id, "reason": "size_mismatch", **_sizes})
    except (OSError, ValueError) as _error:
        _issues.append({"sample_id": _id, "reason": "header_read_error", "error": str(_error)})
    if _number % 1000 == 0:
        print("Scanned", _number, "pairs; issues:", len(_issues), flush=True)
_audit = {"pairs_scanned": len(_index["images"]), "issues": _issues}
_audit_path = DATA / "gta_dimension_audit.json"
_audit_path.write_text(_json.dumps(_audit, indent=2), encoding="utf-8")
print(_json.dumps(_audit, indent=2))
print("Report:", _audit_path)
```

Giữ strict validation ở Cell 10. Nếu sau này quyết định sửa annotation hoặc
loại cặp lỗi, phải có policy/provenance tường minh, tạo lại manifest và tạo run
mới. Checkpoint đã gắn với manifest cũ không thể resume trên split đã thay đổi.

## Trường hợp 15188 và 17705: RGB 957×526, label P 1914×1052

Người dùng đã quan sát RGB đúng nửa kích thước mask theo cả hai chiều. Mode P
là palette-index mask, được đọc trực tiếp thành IDs; không convert label sang
RGB. Giải pháp hỗ trợ là **opt-in cho từng ID**: phóng RGB bằng bilinear đúng
2× khi đọc, giữ nguyên mask và PNG gốc. Cần đối chiếu nội dung bằng overlay;
tỉ lệ kích thước không tự chứng minh hai ảnh cùng geometry. Upsampling không
khôi phục chi tiết đã mất do ảnh RGB được lưu ở độ phân giải thấp.

## Trường hợp 20801: RGB 1914×1046, label 1276×697

Đây không phải RGB half-size. Scale theo chiều rộng là `1276 / 1914 = 2/3`;
chiều cao theo cùng scale là `1046 × 2/3 = 697.333`, chênh nhãn khoảng 0.333 pixel.
Kích thước phù hợp với giả thuyết đổi độ phân giải có làm tròn, nhưng chưa chứng
minh ảnh và nhãn cùng nội dung/hình học. Đối chiếu ZIP và xem overlay trước khi train.

Tùy chọn mới `--gta-resize-image-id 20801` chỉ cho ID được chỉ định, chỉ GTA:
resize **RGB** bilinear về đúng kích thước nhãn 1276×697 khi đọc, giữ nguyên mask
và PNG gốc. Điều kiện kích thước là
`abs(image_height * label_width / image_width - label_height) <= 1` pixel.
Sai số này được ghi trong metadata policy; loader/inspect kiểm tra lại dimensions
và SHA256 hai input. Không convert palette mask sang RGB hoặc nội suy mask bilinear.

## Cập nhật code và chạy lại manifest

Trên local, commit/push code mới (gồm data/manifest.py, dataset.py, inspect.py,
cli.py và tests) cùng các thay đổi epoch logging nếu chưa push:

```bash
cd /f/CRISP_DA
git add src/crisp_da configs tests README_B0.md B0_VALIDATION.md EPOCH_EVAL_LOGGING.md GTA_SIZE_MISMATCH_DIAGNOSTIC.md MOLAB_B0_GTA_01_03_RUNBOOK.md MOLAB_B0_GTA_01_10_RUNBOOK.md
git commit -m "Support audited GTA RGB alignment with rounding"
git push origin main
```

Trên MoLab, dùng
chế độ manual và chạy lại Cell 1–2 mới để checkout `main` thành SHA đầy đủ và
cài package. Cell 1 cập nhật cả biến GIT_REF cho metadata/snapshot; không chỉ
checkout Git ở cell khác rồi để GIT_REF cũ. Giữ DATA/ARCHIVES, không cần tải và
giải nén lại. Sau đó tạo manifest:

```python
run_crisp(
    "make-manifests", "--data-root", DATA,
    "--gta-encoding", "raw_ids", "--city-encoding", "raw_ids",
    "--dev-fraction", "0.05", "--seed", "42",
    "--gta-upsample-image-id", "15188",
    "--gta-upsample-image-id", "17705",
    "--gta-resize-image-id", "20801",
)
```

Cell 10 của runbook full đã dùng danh sách public GTA_UPSAMPLE_IMAGE_IDS từ Cell 1.
Danh sách đã biết là `["15188", "17705"]`. **Chạy Cell 9a mới** để quét header
toàn GTA, lấy GTA_DETECTED_UPSAMPLE_IDS, rồi Cell 10 hợp nhất hai danh sách và
truyền từng ID vào CLI. Như vậy không phải đợi full decode dừng ở từng ID mới.
Report ghi ở DATA/gta_dimension_audit.json, được sao vào RUN_DIR khi chuẩn bị
train để đi cùng snapshot. Cell này không sửa hay xóa ảnh. Ngoài half-size,
Cell 9a mới gom GTA_DETECTED_RESIZE_IDS cho các cặp cùng tỷ lệ khung hình với
sai số làm tròn tối đa 1 pixel chiều cao đích. Cell 10 truyền danh sách này vào
--gta-resize-image-id. Tỷ lệ khác vượt ngưỡng/header lỗi vẫn chặn; Cityscapes vẫn strict.

**Trường hợp 20801 cần core code mới có --gta-resize-image-id**, không thể chỉ thêm
20801 vào danh sách upsample cũ. Commit/push, thay Cell 1–2 và chạy lại để checkout
code mới; sau đó thay/chạy Cell 9a–10. Không cần tải/giải nén lại dữ liệu đã có.
Không định nghĩa lại các biến public của Cell 1 trong cell mới; danh sách phát
hiện dùng public variables riêng do Cell 9a tạo.

Mỗi row ngoại lệ trong JSONL có `image_alignment` với policy, original/target
dimensions và SHA256 cả ảnh/nhãn. `report.json` liệt kê `image_alignments`.
Loader/inspect áp dụng cùng một policy và từ chối khi bytes input đã thay đổi.
Không bỏ ảnh, không sửa archive, không làm thay đổi palette indices trong mask.

Cell 11 mới chia nhóm tối đa 8 ngoại lệ để xuất tất cả overlay vào
`inspection/aligned_source/batch_*/`, kể cả khi ID không nằm trong 8 ảnh đầu
của source train. Xem overlay riêng các mẫu `15188`, `17705`, `20801` và ID phát hiện thêm
rồi mới xác nhận MAPPING_VERIFIED=True. Snapshot lưu cả manifest có policy.

Manifest cũ không có policy không tự đổi hành vi. Nếu đã có checkpoint, không
resume nó với manifests mới; chọn run mới. Khi resume run có policy, giữ nguyên
danh sách ID và hash manifest. Với run 01–03 không có ID15188, không thêm flag này.
