# MoLab: train B0 MiT-B5 source-only trên GTA phần 01–03

Ngày: 18/10/2026. Repository: <https://github.com/trungphan27/CRISP-DA>.

Tài liệu gồm các cell **Python cho marimo/MoLab**, chạy theo thứ tự từ một
runtime mới. Code core vẫn nằm trong repository. Không dán các lệnh Git Bash,
`!pip` hay `export` vào các cell này.

## Phạm vi và những điểm cần biết

- Train trên `01_images.zip`, `02_images.zip`, `03_images.zip` và **ba ZIP nhãn
  tương ứng**. Đây là ba phần của cùng dataset GTA, không phải ba dataset độc lập.
  Nếu mỗi phần có 2.500 cặp thì tổng là 7.500; cell sẽ đếm và kiểm tra thực tế.
- Source train/dev chia theo ảnh, seed 42, dev 5%; với 7.500 ảnh sẽ là 7.125/375.
- Cityscapes dùng **500 ảnh val + labelIds để đánh giá**. Không đọc target train
  GT, không dùng pseudo-label, EMA teacher, prototype hoặc target loss.
- MiT-B5 encoder pretrained ImageNet, decoder của project, CE 19 lớp,
  `ignore_index=255`; riêng GTA raw ID 34 cũng được chuyển thành 255.
- Mặc định run thử **1.000 optimizer steps**, crop 512×512, batch 2. Đây là budget
  thử trên subset, **không phải khẳng định đã hội tụ hoặc benchmark GTA đầy đủ**.
  Bạn có thể sửa budget trước khi bắt đầu run chính, sau khi đo smoke timing.
- Mỗi đoạn 100 optimizer steps: trainer lưu `last.pt` rồi thoát; notebook đóng
  gói checkpoint + manifests + config + log, upload Drive, tải lại và so SHA256.
  Chỉ khi xác minh thành công mới chạy đoạn tiếp. LR schedule vẫn theo toàn bộ
  `max_steps`, không khởi động lại sau mỗi đoạn.
- Nếu runtime mất giữa một đoạn, chỉ bảo đảm khôi phục tới **snapshot gần nhất
  đã xác minh**. Nếu chỉ trình duyệt mất kết nối, kiểm tra runtime trước khi chạy
  thêm một trainer; không khởi chạy đồng thời hai tiến trình trên cùng RUN_DIR.
- ZIP tải bằng code, dữ liệu giải nén và checkpoint local là scratch. Source,
  `.env` và file được upload qua Files có chính sách lưu khác; không dựa vào
  giới hạn 1 GB để kết luận checkpoint tự được giữ. [MoLab storage](https://marimo.io/pages/molab/storage).
- Bộ ZIP cần tải khoảng 28–29 GiB; phần giải nén chọn lọc khoảng 18–19 GiB.
  Cộng model cache, optimizer checkpoint, snapshot và bản readback, nên dự trù
  **ít nhất khoảng 55–60 GiB workspace**. Con số chính xác in từ ZIP.
  Overlay báo 8 EiB không chứng minh quota; tải/giải nén vẫn có thể hết đĩa.
- Mỗi snapshot B5 có thể cỡ 1 GiB. Mặc định giữ tất cả snapshot trên Drive;
  10 đoạn có thể cần hơn 10 GiB, cộng snapshot cuối. Kiểm tra dung lượng Drive.

## Cách chạy lần đầu / sau khi mất runtime

**Lần đầu:** giữ `MODE = "new"` ở cell 1. Chạy 1–13; xem overlay rồi xác nhận
mapping ở cell 11. Bấm smoke ở cell 13, xem timing. Chạy tiếp 14–18.

**Resume sau khi runtime mất:** cell 1 chọn `MODE = "resume"`, giữ cùng RUN_ID,
điền đường dẫn snapshot và SHA256 đã lưu. Chạy 1–12; cell 7 khôi phục và cell 10
so sánh manifest. Cell 13 bỏ qua smoke; cell 14 lấy nguyên config từ checkpoint
đã sao lưu. Sau đó 15–18. Dataset vẫn phải có lại ở runtime mới.

Trong MoLab nên dùng chế độ chạy thủ công cho notebook có tác vụ dài. Các cell
download và extraction có kiểm tra file cũ. Hai cell training có **Run button**;
không dùng Run All để thay cho việc đọc kết quả từng bước.

## Cell 1 — Đường dẫn, chế độ và clone đúng phiên bản code

Chỉ đổi RUN_ID khi bắt đầu một thử nghiệm mới. Phiên bản dưới đây có fix GTA 34.
Giữ cùng Git commit khi resume. Cell dừng nếu checkout có thay đổi chưa commit.

```python
import os
import sys
import json
import hashlib
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

MODE = "new"                 # "new" hoặc "resume"
RUN_ID = "b0-gta01-03-trial-001"
RESUME_REMOTE = ""           # Resume: đường dẫn .tar trên Drive
RESUME_SHA256 = ""           # Resume: SHA256 từ dòng VERIFIED / sidecar
GIT_REF = "main"             # New: main; resume: SHA đầy đủ trong snapshot
REPO = Path("/marimo/CRISP-DA")
WORK = Path("/marimo/crisp-work")
DATA = WORK / "data-gta01-03"
CACHE = WORK / "cache/huggingface"
ARCHIVES = WORK / "cache/archives"
RUN_DIR = WORK / "runs" / RUN_ID
DRIVE_DATA = "UDASS_PDT/Project III/CRISP-DA/dataset"
# Có thể đổi sang thư mục Drive do chính bạn sở hữu nếu thiếu quyền ghi.
DRIVE_BACKUPS = "UDASS_PDT/Project III/CRISP-DA/outputs/b0_subset_runs"

assert MODE in {"new", "resume"}
assert RUN_ID and Path(RUN_ID).name == RUN_ID
if MODE == "resume":
    assert RESUME_REMOTE and len(RESUME_SHA256) == 64
for _path in (WORK, DATA, CACHE, ARCHIVES, RUN_DIR):
    _path.mkdir(parents=True, exist_ok=True)
if not REPO.exists():
    subprocess.run([
        "git", "clone", "https://github.com/trungphan27/CRISP-DA.git", str(REPO)
    ], check=True)
_dirty = subprocess.check_output(
    ["git", "status", "--porcelain"], cwd=REPO, text=True
).strip()
if _dirty:
    raise RuntimeError("Checkout có thay đổi local; kiểm tra trước khi đổi commit:\n" + _dirty)
if GIT_REF == "main":
    subprocess.run(["git", "fetch", "origin", "main"], cwd=REPO, check=True)
    GIT_REF = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=REPO, text=True
    ).strip()
subprocess.run(["git", "checkout", "--detach", GIT_REF], cwd=REPO, check=True)
print("Git:", subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip())
print("MODE:", MODE, "RUN_DIR:", RUN_DIR)
```

## Cell 2 — Cài package vào đúng Python của notebook

Nếu báo thiếu `crisp_da.engine.history`, code mới chưa có trong checkout MoLab.
Commit/push các thay đổi local gồm cả file mới `src/crisp_da/engine/history.py`,
rồi chạy lại Cell 1 (run mới: GIT_REF="main") và Cell 2 ở chế độ manual.
Cài lại pip không tải được file chưa commit. Không cần tải lại dataset.
MODE=resume vẫn dùng đúng SHA đã lưu, không tự đổi phiên bản code giữa run.

Giữ PyTorch GPU đã chạy tốt. Không thêm `--upgrade` toàn bộ dependencies.
Thông báo có pip mới không bắt buộc phải nâng cấp. Nếu environment không có pip,
hãy dùng trình quản lý package của MoLab để cài project vào cùng interpreter.

```python
_history_file = REPO / "src/crisp_da/engine/history.py"
if not _history_file.is_file():
    raise RuntimeError(
        "Checkout MoLab thiếu src/crisp_da/engine/history.py. "
        "Local: git add src/crisp_da configs tests; commit và push. "
        "MoLab run mới: Cell 1 GIT_REF='main', chạy lại Cell 1 rồi Cell 2. "
        "Không cần tải lại dataset."
    )
subprocess.run([
    sys.executable, "-m", "pip", "install", "-e", str(REPO)
], check=True)
subprocess.run([
    sys.executable, "-m", "crisp_da.cli", "--help"
], cwd=REPO, check=True)
subprocess.run([
    sys.executable, "-c",
    "import importlib.util; assert importlib.util.find_spec('crisp_da.engine.history'), "
    "'Checkout chua co epoch eval history: commit/push code moi len GitHub truoc'",
], cwd=REPO, check=True)
```

## Cell 3 — Helper và kiểm tra GPU/RAM/đĩa

Nếu GPU không hoạt động, chọn lại GPU runtime trước khi đi tiếp. CUDA driver và
CUDA build của PyTorch không cần cùng số phiên bản; phép forward/backward thật
trong smoke mới kiểm tra model B5 trên runtime này.

```python
def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()

def run_crisp(*args, overrides=None):
    env = os.environ.copy()
    env.update(DATA_ROOT=str(DATA), CACHE_DIR=str(CACHE), RUN_DIR=str(RUN_DIR))
    if overrides:
        env.update({key: str(value) for key, value in overrides.items()})
    subprocess.run([
        sys.executable, "-u", "-m", "crisp_da.cli", *map(str, args)
    ], cwd=REPO, env=env, check=True)

subprocess.run(["nvidia-smi"], check=True)
subprocess.run([
    sys.executable, "-c",
    "import torch; print('torch', torch.__version__, 'CUDA build', torch.version.cuda); "
    "assert torch.cuda.is_available(), 'CUDA unavailable'; "
    "print(torch.cuda.get_device_name(0)); "
    "print('BF16', torch.cuda.is_bf16_supported()); "
    "print('free,total GiB', [v/1024**3 for v in torch.cuda.mem_get_info()]); "
    "x=torch.randn(64,64,device='cuda',requires_grad=True); "
    "(x@x).sum().backward(); torch.cuda.synchronize(); print('GPU PASS')"
], check=True)
subprocess.run(["free", "-h"], check=True)
for _name in ("memory.max", "memory.current"):
    _cgroup = Path("/sys/fs/cgroup") / _name
    if _cgroup.exists():
        print(_name, _cgroup.read_text().strip())
print("Disk report only (may not reflect quota):", shutil.disk_usage(WORK))
```

## Cell 4 — Kết nối Drive: chạy cell do MoLab sinh

Mở **Files → Add remote storage → Google Drive**. Để trống Service Account JSON
để đăng nhập bằng trình duyệt; giữ `fsspec`. Chạy cell kết nối mà MoLab tạo và
hoàn thành OAuth. Dùng tài khoản có quyền đọc dataset **và quyền ghi backups**.

Cell được tạo thường tương tự đoạn dưới. **Giữ snippet thực tế của MoLab** nếu
có thêm cấu hình xác thực. Chỉ định nghĩa `fs` ở một cell; nếu biến có tên khác,
đổi tham chiếu trong các cell sau cho thống nhất. Không dùng `google.colab`.

```python
from gdrive_fsspec import GoogleDriveFileSystem

fs = GoogleDriveFileSystem(
    use_listings_cache=False,
    skip_instance_cache=True,
)
```

[MoLab remote storage và OAuth](https://marimo.io/blog/drive-storage).
Đường dẫn dưới đây lấy từ snippet thực tế bạn đã cung cấp; không thay bằng URL
chia sẻ hoặc file ID. Nếu tài khoản nhìn thấy cây khác, dùng **Copy path**.

## Cell 5 — Test đọc Drive và upload/readback trước khi tải dataset lớn

Chỉ PASS khi đọc được file dataset và ghi/đọc được một file thử trong backups.
Nếu báo permission denied, đổi DRIVE_BACKUPS sang một thư mục bạn có quyền ghi;
không đổi quyền chia sẻ dataset. File thử nhỏ được giữ lại để kiểm tra.

```python
import uuid

print(fs.info(f"{DRIVE_DATA}/gta/labels/01_labels.zip"))
fs.makedirs(DRIVE_BACKUPS, exist_ok=True)
_probe = WORK / "drive-probe.txt"
_probe.write_text("CRISP-DA Drive read/write test\n", encoding="utf-8")
_probe_remote = f"{DRIVE_BACKUPS}/probe-{uuid.uuid4().hex}.txt"
fs.put(str(_probe), _probe_remote)
_probe_readback = WORK / "drive-probe-readback.txt"
fs.get(_probe_remote, str(_probe_readback))
assert file_sha256(_probe) == file_sha256(_probe_readback)
print("PASS: Drive read + write + SHA256 readback")
```

## Cell 6 — Khai báo snapshot helper và hàm khôi phục an toàn

Snapshot chứa full training state trong `last.pt`, run artifacts, exact manifest,
encoder lock và provenance của dataset. Không chứa ảnh, ZIP hoặc credentials.
Không nén checkpoint để tránh tốn CPU; tên remote duy nhất, không ghi đè bản tốt
trước. Upload xong tải lại toàn bộ archive, so SHA256 rồi ghi sidecar `.json`.
Nếu upload/readback lỗi, cell train dừng tại checkpoint local hiện có.

```python
import tarfile

def verified_snapshot(run_path, tag):
    run_path = Path(run_path)
    assert (run_path / "last.pt").is_file(), "Chưa có checkpoint"
    summary = json.loads((run_path / "summary.json").read_text())
    step = summary["step"]
    token = uuid.uuid4().hex[:12]
    stem = f"{tag}-step-{step:07d}-{token}"
    local_archive = WORK / f"{stem}.tar"
    readback = WORK / f"{stem}.readback.tar"
    remote_dir = f"{DRIVE_BACKUPS}/{RUN_ID}"
    remote = f"{remote_dir}/{stem}.tar"
    fs.makedirs(remote_dir, exist_ok=True)
    with tarfile.open(local_archive, "w") as bundle:
        bundle.add(run_path, arcname="run")
        bundle.add(DATA / "manifests", arcname="manifests")
        bundle.add(REPO / "configs/mit_b5_imagenet.lock.json", arcname="encoder.lock.json")
        bundle.add(WORK / "dataset_provenance.json", arcname="dataset_provenance.json")
    digest = file_sha256(local_archive)
    print("Uploading:", remote, flush=True)
    fs.put(str(local_archive), remote)
    print("Readback verification...", flush=True)
    fs.get(remote, str(readback))
    if file_sha256(readback) != digest:
        raise RuntimeError("Snapshot readback SHA256 mismatch; chưa được coi là backup tốt")
    metadata = {
        "snapshot": remote, "sha256": digest, "step": step, "tag": tag,
        "run_id": RUN_ID, "git_commit": GIT_REF,
        "manifest_sha256": {
            p.name: file_sha256(p) for p in sorted((DATA / "manifests").glob("*.jsonl"))
        },
        "created_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
    }
    local_meta = WORK / f"{stem}.json"
    local_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    fs.put(str(local_meta), remote + ".json")
    _meta_readback = WORK / f"{stem}.readback.json"
    fs.get(remote + ".json", str(_meta_readback))
    assert file_sha256(local_meta) == file_sha256(_meta_readback)
    print("VERIFIED SNAPSHOT:", remote, flush=True)
    print("SHA256:", digest, flush=True)
    print("STEP:", step, flush=True)
    # Chỉ xóa các bản tar tạm do hàm này vừa tạo; giữ checkpoint và bản Drive.
    local_archive.unlink()
    readback.unlink()
    return metadata

def restore_snapshot(remote, expected_sha):
    archive = WORK / "restore-snapshot.tar"
    fs.get(remote, str(archive))
    if file_sha256(archive) != expected_sha:
        raise RuntimeError("SHA256 snapshot không khớp; không giải nén")
    destination = WORK / "restored" / uuid.uuid4().hex
    destination.mkdir(parents=True)
    with tarfile.open(archive, "r") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise RuntimeError("Unsafe snapshot path")
            if not (member.isfile() or member.isdir()):
                raise RuntimeError("Snapshot chứa link hoặc special file")
        bundle.extractall(destination, filter="data")
    archive.unlink()
    return destination
```

Snapshot chỉ load từ tài khoản/lưu trữ bạn tin cậy: checkpoint đầy đủ dùng
`torch.load(weights_only=False)` để phục hồi Python/NumPy RNG.

## Cell 7 — Khôi phục nếu MODE=resume

Lấy RESUME_REMOTE/SHA256 từ dòng VERIFIED đã lưu, hoặc file sidecar `.tar.json`
trên Drive (không chọn `.tar` upload dở). Đường dẫn snapshot chính có tag `train`;
snapshot tag `smoke` chỉ dùng kiểm tra, không resume thành run chính.

```python
RESTORED = None
if MODE == "resume":
    if (RUN_DIR / "last.pt").exists():
        raise FileExistsError("RUN_DIR đã có last.pt; không chép đè. Kiểm tra phiên hiện tại.")
    RESTORED = restore_snapshot(RESUME_REMOTE, RESUME_SHA256)
    _saved_cfg = json.loads((RESTORED / "run/config.json").read_text())
    if _saved_cfg["run"]["kind"] != "benchmark":
        raise RuntimeError("Đã chọn smoke checkpoint; hãy chọn snapshot tag train")
    _saved_env = json.loads((RESTORED / "run/environment.json").read_text())
    if _saved_env.get("git_commit") != GIT_REF:
        raise RuntimeError("Git commit khác snapshot; sửa GIT_REF ở cell 1 rồi chạy lại")
    shutil.copytree(RESTORED / "run", RUN_DIR, dirs_exist_ok=True)
    print("Khôi phục checkpoint:", RUN_DIR / "last.pt")
    print("Step:", json.loads((RUN_DIR / "summary.json").read_text())["step"])
else:
    if (RUN_DIR / "last.pt").exists():
        raise FileExistsError("Run này đã có checkpoint. Chọn RUN_ID mới hoặc khôi phục đúng run.")
    print("New run: không khôi phục checkpoint.")
```

## Cell 8 — Download 6 ZIP GTA + 2 ZIP Cityscapes

`fs.get` hiện dùng tải lại từ đầu nếu `.part` bị ngắt. File `.zip` đã có chỉ được
bỏ qua tải khi size và CRC đạt. Mất runtime thì có thể phải tải lại tất cả.
Tải lần lượt; không chạy song song các snippet cũ. Cell sẽ in tổng size trước.

```python
import zipfile

ZIP_JOBS = []
for _part in (1, 2, 3):
    for _kind in ("images", "labels"):
        _name = f"{_part:02d}_{_kind}.zip"
        ZIP_JOBS.append((_name, f"{DRIVE_DATA}/gta/{_kind}/{_name}"))
for _name in ("leftImg8bit_trainvaltest.zip", "gtFine_trainvaltest.zip"):
    ZIP_JOBS.append((_name, f"{DRIVE_DATA}/cityscapes/{_name}"))

_sizes = {name: fs.info(remote).get("size") for name, remote in ZIP_JOBS}
print("Compressed GiB (known sizes):", sum(int(v) for v in _sizes.values() if v is not None)/1024**3)
_records = []
for _name, _remote in ZIP_JOBS:
    _local = ARCHIVES / _name
    _part_path = ARCHIVES / (_name + ".part")
    print("\nZIP:", _name, flush=True)
    if _local.exists():
        _candidate = _local
    else:
        fs.get(_remote, str(_part_path))
        _candidate = _part_path
    _expected = _sizes[_name]
    if _expected is not None and _candidate.stat().st_size != int(_expected):
        raise RuntimeError(f"Size mismatch: {_candidate}")
    print("Checking CRC...", flush=True)
    with zipfile.ZipFile(_candidate) as _zip:
        _bad = _zip.testzip()
        if _bad is not None:
            raise RuntimeError(f"CRC error: {_name}: {_bad}")
    if _candidate == _part_path:
        _part_path.replace(_local)
    _records.append({
        "name": _name, "remote_path": _remote,
        "bytes": _local.stat().st_size, "sha256": file_sha256(_local),
    })
    print("PASS:", _name, flush=True)
_provenance = {"gta_parts": [1, 2, 3], "archives": _records}
if RESTORED is not None:
    _old = json.loads((RESTORED / "dataset_provenance.json").read_text())
    if _old != _provenance:
        raise RuntimeError("Archive provenance thay đổi; không resume trên dữ liệu khác")
(WORK / "dataset_provenance.json").write_text(
    json.dumps(_provenance, indent=2), encoding="utf-8"
)
```

Nếu `.zip` đã tồn tại nhưng bị lỗi, cell dừng và ghi rõ file. Kiểm tra/đổi tên
file lỗi rồi tải lại; không tiếp tục train với archive không đạt.

## Cell 9 — Giải nén có chọn lọc, có thể chạy lại

Giữ layout `images/`, `labels/` trong GTA. Cityscapes chỉ lấy val RGB và semantic
labelIds; không lấy `instanceIds`, ảnh màu nhãn hoặc target train GT.
Từ chối trùng ID giữa các phần. Với file đã giải nén, chỉ skip khi size+CRC khớp
ZIP. File khác nội dung gây lỗi; không ghi đè. Không tự xóa ZIP sau giải nén.

```python
import stat
import zlib
from pathlib import PurePosixPath

def crc32_file(path):
    value = 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            value = zlib.crc32(block, value)
    return value & 0xffffffff

_extract_jobs = []
for _part in (1, 2, 3):
    for _kind in ("images", "labels"):
        _extract_jobs.append((f"{_part:02d}_{_kind}.zip", "gta", f"{_kind}/", ".png"))
_extract_jobs += [
    ("leftImg8bit_trainvaltest.zip", "cityscapes", "leftImg8bit/val/", "_leftImg8bit.png"),
    ("gtFine_trainvaltest.zip", "cityscapes", "gtFine/val/", "_gtFine_labelIds.png"),
]
_seen = set()
_gta_names = {"images": set(), "labels": set()}
_selected_bytes = 0
for _name, _domain, _prefix, _suffix in _extract_jobs:
    _root = (DATA / _domain).resolve()
    with zipfile.ZipFile(ARCHIVES / _name) as _zip:
        _members = [m for m in _zip.infolist() if not m.is_dir()
                    and m.filename.startswith(_prefix) and m.filename.endswith(_suffix)]
        if not _members:
            raise RuntimeError(f"Không có member phù hợp: {_name}")
        print(_name, len(_members), "files", flush=True)
        for _index, _member in enumerate(_members, 1):
            _relative = PurePosixPath(_member.filename)
            if "\\" in _member.filename or _relative.is_absolute() or ".." in _relative.parts:
                raise RuntimeError("Unsafe ZIP path")
            if stat.S_ISLNK(_member.external_attr >> 16):
                raise RuntimeError("ZIP symlink rejected")
            _target = _root.joinpath(*_relative.parts).resolve()
            if not _target.is_relative_to(_root) or _target in _seen:
                raise RuntimeError(f"Duplicate/unsafe target: {_target}")
            _seen.add(_target)
            _selected_bytes += _member.file_size
            if _domain == "gta":
                _kind = _prefix.rstrip("/")
                if _target.name in _gta_names[_kind]:
                    raise RuntimeError(f"Duplicate GTA filename: {_target.name}")
                _gta_names[_kind].add(_target.name)
            if _target.exists():
                if (_target.stat().st_size != _member.file_size
                        or crc32_file(_target) != _member.CRC):
                    raise RuntimeError(f"Existing file differs from ZIP: {_target}")
            else:
                _target.parent.mkdir(parents=True, exist_ok=True)
                _temporary = _target.with_name(_target.name + ".extract-part")
                with _zip.open(_member) as _input, _temporary.open("wb") as _output:
                    shutil.copyfileobj(_input, _output, length=1024**2)
                if (_temporary.stat().st_size != _member.file_size
                        or crc32_file(_temporary) != _member.CRC):
                    raise RuntimeError(f"Extract CRC mismatch: {_temporary}")
                _temporary.replace(_target)
            if _index % 500 == 0:
                print(" ", _index, "/", len(_members), flush=True)
assert _gta_names["images"] == _gta_names["labels"], "GTA image/label mismatch"
for _kind in ("images", "labels"):
    _actual = {p.name for p in (DATA / "gta" / _kind).rglob("*.png")}
    assert _actual == _gta_names[_kind], "DATA chứa ảnh/nhãn GTA ngoài phần 01–03"
print("GTA pairs:", len(_gta_names["images"]))
print("Selected extracted GiB:", _selected_bytes / 1024**3)
```

## Cell 10 — Scan đầy đủ, tạo manifests và kiểm tra resume

Full scan 7.500 ảnh/nhãn có thể mất thời gian. Lỗi ID khác 34 vẫn phải kiểm tra,
không đổi sang `unknown_policy=ignore` để vượt qua lỗi. Không đổi encoding sang
`train_ids` khi đang dùng raw IDs. Full scan xác minh bytes ảnh/nhãn đọc được,
kích thước, giá trị nhãn; overlay ở cell sau kiểm tra ý nghĩa mapping.
Cell này tự đếm PNG từ DATA, không dùng `_gta_names` của Cell 9: trong marimo,
biến bắt đầu bằng `_` chỉ có phạm vi trong cell khai báo nó.

```python
run_crisp(
    "make-manifests", "--data-root", DATA,
    "--gta-encoding", "raw_ids", "--city-encoding", "raw_ids",
    "--dev-fraction", "0.05", "--seed", "42",
)
_report = json.loads((DATA / "manifests/report.json").read_text())
for _split, _entry in _report["manifests"].items():
    print(_split, _entry)
assert _report["manifests"]["target_val"]["count"] == 500
_source_png_count = sum(1 for _p in (DATA / "gta/images").rglob("*.png"))
_source_manifest_count = sum(
    _report["manifests"][_key]["count"]
    for _key in ("source_train", "source_dev")
)
assert _source_manifest_count == _source_png_count, (
    f"Source manifest có {_source_manifest_count} ảnh, "
    f"nhưng thư mục GTA images có {_source_png_count} PNG"
)
if RESTORED is not None:
    for _path in (RESTORED / "manifests").glob("*.jsonl"):
        assert file_sha256(DATA / "manifests" / _path.name) == file_sha256(_path), "Manifest changed"
    print("PASS: regenerated manifests match saved snapshot")
```

## Cell 11 — Kiểm tra overlay và xác nhận mapping

Inspect 32 ảnh mỗi split để xem histogram; tiện ích hiện xuất tối đa 8 ảnh mỗi
lần gọi. Đây là kiểm tra mẫu, không phải xem thủ công toàn dataset. Bộ ảnh hiển
thị gồm **RGB | mask remapped | overlay**; kiểm tra road/sky/car/person đúng vị trí.
Nếu chưa đủ tin cậy, inspect thêm các nhóm ảnh bằng manifest tạm riêng trước
khi xác nhận. Không tự chỉnh các manifest chính của run đang resume.

```python
for _split in ("source_train", "target_val"):
    run_crisp(
        "inspect-data", "--data-root", DATA,
        "--manifest", DATA / f"manifests/{_split}.jsonl",
        "--output", WORK / f"inspection/{_split}", "--limit", "32",
    )
```

Cell hiển thị ảnh:

```python
import marimo as mo

mo.vstack([
    mo.image(src=str(_p), caption=str(_p.relative_to(WORK)))
    for _split in ("source_train", "target_val")
    for _p in sorted((WORK / "inspection" / _split).glob("sample_*.png"))[:4]
])
```

Cell xác nhận: sau khi xem ảnh và report, đổi `False` thành `True` rồi chạy.
Nếu mapping sai, sửa nguyên nhân trước; không dùng checkbox để che lỗi.

```python
MAPPING_VERIFIED = False  # ĐỔI THÀNH True sau khi đã kiểm tra
assert MAPPING_VERIFIED, "Cần kiểm tra overlay/mapping trước khi train"
```

## Cell 12 — Tải encoder ImageNet đã pin và so checksum

Chỉ tải ~328 MB weights encoder, không dùng segmentation checkpoint Cityscapes.
Không ghi đè file lock trong Git. Lock của bản tải được so với metadata đã commit.

```python
_download_lock = WORK / "downloaded_encoder.lock.json"
run_crisp("lock-encoder", "--cache-dir", CACHE, "--output", _download_lock)
_expected_lock = json.loads((REPO / "configs/mit_b5_imagenet.lock.json").read_text())
_actual_lock = json.loads(_download_lock.read_text())
assert _actual_lock == _expected_lock, "Encoder lock mismatch"
print("PASS: ImageNet MiT-B5 revision + SHA256")
```

## Cell 13 — Smoke 20 steps, timing, GPU thật (new run)

Tạo config ngoài Git để giữ checkout sạch. Smoke không thay đổi config budget
của run chính. Eval toàn source dev + 500 Cityscapes val ở cuối có thể lâu hơn
20 train steps. Resume không cần chạy lại smoke trừ khi đổi môi trường có vấn đề.

Cell tạo nút, rồi cell tiếp theo thực hiện khi bấm nút:

```python
smoke_button = mo.ui.run_button(label="Chạy B5 smoke 20 steps (new run)")
smoke_button
```

```python
mo.stop(MODE == "resume" or not smoke_button.value)
import yaml as _yaml
import time as _time

_smoke_cfg = _yaml.safe_load((REPO / "configs/b0_source_only.yaml").read_text())
_smoke_dir = WORK / "runs" / (RUN_ID + "-smoke")
if (_smoke_dir / "last.pt").exists():
    raise FileExistsError("Smoke đã chạy; dùng kết quả có sẵn, không train đè")
_smoke_cfg["run"] = {"kind": "smoke", "budget_note": "B5 GTA01-03 preflight; not a research baseline"}
_smoke_cfg["paths"] = {"data_root": str(DATA), "cache_dir": str(CACHE), "run_dir": str(_smoke_dir)}
_smoke_cfg["model"].update(pretrained_lock=str(REPO / "configs/mit_b5_imagenet.lock.json"), local_files_only=True)
_smoke_cfg["data"]["mapping_verified"] = MAPPING_VERIFIED
_smoke_cfg["train"].update(max_steps=20, warmup_steps=2, log_every=1, eval_every=20, eval_every_epoch=False, checkpoint_every=20)
_smoke_path = WORK / "smoke.yaml"
_smoke_path.write_text(_yaml.safe_dump(_smoke_cfg, sort_keys=False))
_started = _time.monotonic()
run_crisp("train", "--config", _smoke_path, overrides={"RUN_DIR": _smoke_dir})
_elapsed = _time.monotonic() - _started
_events = [json.loads(line) for line in (_smoke_dir / "metrics.jsonl").read_text().splitlines()]
_steps = [r for r in _events if r["event"] == "train"]
_steady = _steps[5:] or _steps
print("Mean step seconds (exclude first 5):", sum(r["seconds_per_step"] for r in _steady)/len(_steady))
print("Peak allocated VRAM GiB:", max(r["peak_vram_gib"] for r in _steps))
print("Smoke total seconds (includes startup/eval/checkpoint):", _elapsed)
print("Eval:", [r for r in _events if r["event"] == "eval"])
verified_snapshot(_smoke_dir, "smoke")
```

Nếu smoke đã có `last.pt` do bấm lại, xem các file bên trên thay vì chạy cell
train smoke lần nữa. Nếu OOM, sửa crop/batch trong smoke config, chọn thư mục
smoke mới; sau khi đạt mới khóa cùng thông số cho run chính ở cell 14.
Không dùng mIoU của 20 steps để chọn cấu hình tối ưu theo target val.

## Cell 14 — Khóa config run thử trên subset / giữ nguyên config khi resume

Mặc định 1.000 steps chỉ là thử nghiệm ban đầu (batch 2 tương đương khoảng
2.000 lượt ảnh, chưa đủ một epoch nếu train có 7.125 ảnh). Bạn sửa budget/crop/
batch/precision **tại đây trước run mới**, sau khi xem timing và VRAM.

Ở resume, các thông số dưới không được áp dụng lại: lấy nguyên config đã lưu,
chỉ đổi paths. Không tăng `max_steps` của checkpoint đang resume vì sẽ đổi LR
schedule. Muốn đổi budget tạo experiment khác và ghi rõ cách khởi tạo.

```python
import yaml

CONFIG_PATH = WORK / "configs" / f"{RUN_ID}.yaml"
CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
if MODE == "new":
    _smoke_dir = WORK / "runs" / (RUN_ID + "-smoke")
    _smoke_summary = json.loads((_smoke_dir / "summary.json").read_text())
    assert _smoke_summary["training_finished"], "Smoke chưa hoàn tất"
    assert MAPPING_VERIFIED
    _events = [json.loads(line) for line in (_smoke_dir / "metrics.jsonl").read_text().splitlines()]
    _times = [r["seconds_per_step"] for r in _events if r["event"] == "train"][5:]
    _mean_step = sum(_times)/len(_times)
    _cfg = yaml.safe_load((REPO / "configs/b0_source_only.yaml").read_text())
    _cfg["run"] = {
        "kind": "benchmark",
        "budget_note": (
            f"Exploratory B0 on GTA parts 01-03 only; fixed-step subset trial, "
            f"not full GTA benchmark. Smoke steady mean {_mean_step:.4f} s/step. "
            "Chosen initial budget: 1000 optimizer steps; include evaluation and Drive I/O overhead."
        ),
    }
    _cfg["train"].update(
        max_steps=1000, warmup_steps=100,
        batch_size=2, grad_accumulation=1, workers=0,
        log_every=20, eval_every=0, eval_every_epoch=True, checkpoint_every=100,
        save_best_target_val=False,
    )
    _cfg["data"].update(mapping_verified=True, crop_hw=[512, 512], eval_hw=[512, 1024])
else:
    _cfg = json.loads((RUN_DIR / "config.json").read_text())
    _saved_env = json.loads((RUN_DIR / "environment.json").read_text())
    from importlib.metadata import version as _version
    _differences = {k: (_v, _version(k)) for k, _v in _saved_env["packages"].items()
                    if _v is not None and _version(k) != _v}
    if _differences:
        raise RuntimeError(f"Dependencies khác checkpoint; phục hồi đúng versions trước: {_differences}")
    _precision_now = subprocess.check_output([
        sys.executable, "-c",
        "from crisp_da.engine.train import resolve_device,resolve_precision; import sys; "
        "print(resolve_precision(sys.argv[1],resolve_device(sys.argv[2])))",
        _cfg["train"]["precision"], _cfg["train"]["device"],
    ], cwd=REPO, text=True).strip()
    if _precision_now != _saved_env["resolved_precision"]:
        raise RuntimeError("Resolved precision khác checkpoint; cần runtime hỗ trợ cùng precision")
_cfg["paths"] = {"data_root": str(DATA), "cache_dir": str(CACHE), "run_dir": str(RUN_DIR)}
_cfg["model"].update(pretrained_lock=str(REPO / "configs/mit_b5_imagenet.lock.json"), local_files_only=True)
_new_text = yaml.safe_dump(_cfg, sort_keys=False)
if MODE == "new" and CONFIG_PATH.exists() and CONFIG_PATH.read_text() != _new_text:
    raise RuntimeError("Config này đã được khóa và khác nội dung mới. Chọn RUN_ID mới nếu đổi experiment.")
CONFIG_PATH.write_text(_new_text, encoding="utf-8")
# Validate bằng subprocess để config không vô tình nhận env override cũ.
_env = os.environ.copy()
for _key in ("DATA_ROOT", "CACHE_DIR", "RUN_DIR"):
    _env.pop(_key, None)
subprocess.run([
    sys.executable, "-c",
    "from crisp_da.config import load_config; import sys; print(load_config(sys.argv[1]))",
    str(CONFIG_PATH),
], cwd=REPO, env=_env, check=True)
print("Locked config:", CONFIG_PATH)
```

Tên `kind=benchmark` là chế độ validation của CLI hiện tại; `budget_note` ghi
rõ đây là **subset trial**, không biến run này thành kết quả full GTA chính thức.
Khi sửa budget mới nhớ sửa cả nội dung budget_note cho đúng.

## Cell 15 — Ghi environment và metadata trước train

```python
_freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
(RUN_DIR / "requirements.runtime.txt").write_text(_freeze, encoding="utf-8")
shutil.copy2(CONFIG_PATH, RUN_DIR / "run_config.yaml")
shutil.copy2(WORK / "dataset_provenance.json", RUN_DIR / "dataset_provenance.json")
_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
assert _commit == GIT_REF
(RUN_DIR / "run_notes.json").write_text(json.dumps({
    "run_id": RUN_ID, "git_commit": _commit, "gta_parts": [1, 2, 3],
    "setting": "source-only; target val for evaluation only",
    "selection": "fixed optimizer step; save_best_target_val=false",
}, indent=2), encoding="utf-8")
print("Ready:", RUN_DIR)
```

## Cell 16 — Train theo đoạn + backup đã xác minh trước khi chạy tiếp

Tạo nút ở một cell, chạy runner ở cell kế tiếp. Mỗi đoạn 100 steps có overhead
khởi tạo/load optimizer và upload/readback khoảng GiB; đây là đánh đổi để có
checkpoint bền vững thường xuyên với code hiện tại. Đây không phải stream
training liên tục với uploader chạy ngầm. `--stop-after` dùng **step tuyệt đối**;
resume khôi phục optimizer/scheduler/RNG/sampler và không đổi `max_steps`.

```python
train_button = mo.ui.run_button(label="Train B0 GTA01-03 + verified Drive backups")
train_button
```

```python
mo.stop(not train_button.value)

_final_cfg = yaml.safe_load(CONFIG_PATH.read_text())
_max_steps = _final_cfg["train"]["max_steps"]
_interval = _final_cfg["train"]["checkpoint_every"]
_last = RUN_DIR / "last.pt"
_completed = 0
if _last.exists():
    _completed = json.loads((RUN_DIR / "summary.json").read_text())["step"]
    # Khôi phục hoặc chạy lại cùng cell: sao lưu checkpoint hiện có trước.
    verified_snapshot(RUN_DIR, "train")
while _completed < _max_steps:
    _end = min(_completed + _interval, _max_steps)
    _args = ["train", "--config", CONFIG_PATH, "--stop-after", str(_end)]
    if _last.exists():
        _args += ["--resume", _last]
    print(f"\nTRAIN {_completed} -> {_end}/{_max_steps}", flush=True)
    run_crisp(*_args)
    _summary = json.loads((RUN_DIR / "summary.json").read_text())
    assert _summary["step"] == _end
    _backup = verified_snapshot(RUN_DIR, "train")
    _completed = _end
    print("Durable step:", _completed, flush=True)
print("TRAIN FINISHED. Local checkpoint:", _last)
```

Không bấm lại nút khi trainer còn chạy. Nếu upload lỗi nhưng runtime còn sống,
khắc phục kết nối/quota Drive rồi bấm chạy lại: runner đọc `last.pt` và backup
trước khi tiếp tục. Nếu lỗi trainer giữa chừng, không tự suy step từ tên file;
đọc checkpoint thực tế trước khi tiếp tục. `summary.json` chỉ là báo cáo của
đoạn kết thúc sạch, không thay thế kiểm tra checkpoint.

Nếu runtime mất: chỉ resume từ snapshot đã có sidecar và hash verified.
Checkpoint hiện chứa model/optimizer/scheduler/scaler, global step, config,
sampler cursor, RNG Python/NumPy/CPU/CUDA và pretrained provenance.

## Cell 17 — Evaluate độc lập từ checkpoint cuối, xem kết quả

Run chính đã tự lưu `eval_history.csv` và `eval_history.jsonl` sau mỗi source
epoch, xuyên các đoạn resume. Mặc định tắt eval theo steps, vẫn đánh giá cuối
max_steps. Với 7.125 ảnh train và batch 2, 1.000 steps chưa hết một epoch nên
history chỉ có điểm final ở epoch phân số. Xem [vẽ eval curve](EPOCH_EVAL_LOGGING.md).

Standalone evaluation tải model từ checkpoint, không tải lại encoder weights.
Toàn bộ target val được đánh giá ở eval_hw cố định, upsample logits về GT gốc;
không resize GT, không TTA. IoU trong JSON là 0..1, hiển thị % bằng nhân 100.

```python
run_crisp(
    "evaluate", "--checkpoint", RUN_DIR / "last.pt",
    "--data-root", DATA, "--split", "target_val",
    "--output", RUN_DIR / "standalone-target-val",
)
_metrics = json.loads((RUN_DIR / "standalone-target-val/metrics.json").read_text())
print("Samples:", _metrics["samples"])
print("mIoU (%):", None if _metrics["miou"] is None else 100 * _metrics["miou"])
print("Class IoU (%):", {k: None if v is None else 100*v for k, v in _metrics["per_class_iou"].items()})
_steps = json.loads((RUN_DIR / "summary.json").read_text())["step"]
_in_train = json.loads((RUN_DIR / f"eval/step_{_steps}/target_val/metrics.json").read_text())
assert _metrics["confusion_matrix"] == _in_train["confusion_matrix"], "Eval khác checkpoint cuối; cần kiểm tra"
print("PASS: standalone evaluation matches final in-training confusion matrix")
```

Cell xem loss và prediction:

```python
mo.vstack([
    mo.Html((RUN_DIR / "loss_curve.svg").read_text()) if (RUN_DIR / "loss_curve.svg").exists() else mo.md("Không có curve đoạn đầu"),
    *[mo.image(src=str(_p), caption=_p.name)
      for _p in sorted((RUN_DIR / "standalone-target-val").glob("*_rgb_gt_prediction.png"))],
])
```

Do trainer ghi log riêng khi resume, `loss_curve.svg` chỉ biểu diễn đoạn đầu;
`loss_curve_resume.svg` chỉ đoạn resume cuối. Cell sau ghép log mọi đoạn để
xuất curve toàn run (nếu một step có nhiều record, lấy record xuất hiện sau).

```python
_records_by_step = {}
for _log in [RUN_DIR / "metrics.jsonl", *sorted(RUN_DIR.glob("metrics_resume_*.jsonl"))]:
    if _log.exists():
        for _line in _log.read_text().splitlines():
            _record = json.loads(_line)
            if _record.get("event") == "train":
                _records_by_step[_record["step"]] = _record
_merged_log = WORK / "merged-loss.json"
_merged_log.write_text(json.dumps([_records_by_step[k] for k in sorted(_records_by_step)]))
subprocess.run([
    sys.executable, "-c",
    "import json,sys; from crisp_da.engine.train import write_loss_curve; "
    "write_loss_curve(sys.argv[2],json.load(open(sys.argv[1])))",
    str(_merged_log), str(RUN_DIR / "loss_curve_all.svg"),
], cwd=REPO, check=True)
mo.Html((RUN_DIR / "loss_curve_all.svg").read_text())
```

## Cell 18 — Backup cuối gồm evaluation và curve toàn run

```python
FINAL_BACKUP = verified_snapshot(RUN_DIR, "train-final")
print(json.dumps(FINAL_BACKUP, indent=2))
```

**Lưu lại ở nơi ngoài runtime:** đường dẫn `snapshot`, `sha256`, `run_id`,
`git_commit` và `step`. Các giá trị cũng nằm trong sidecar `.tar.json` trên Drive.
Không chỉ lưu chúng trong một biến Python. Trên Drive, mở thư mục
`outputs/b0_subset_runs/<RUN_ID>/` để kiểm tra archive và sidecar.

## Khôi phục vào ngày khác: checklist ngắn

1. Mở notebook đã lưu, chọn GPU. Không chạy lại tất cả cell training tự động.
2. Cell 1: MODE=resume, cùng RUN_ID/GIT_REF, điền snapshot và SHA256 đã xác minh.
3. Cài package, kiểm tra GPU; kết nối lại Drive. Test quyền đọc/ghi.
4. Khôi phục snapshot (cell 7); tải/giải nén data phần 01–03 và Cityscapes val.
5. Regenerate manifest, so hash với bản backup; kiểm tra overlay/mapping.
6. Tải encoder cache nếu thiếu; cell 14 dùng nguyên saved config và kiểm tra
   dependency versions. Không sửa budget/precision/crop/augmentation để resume.
7. Cell 16: trainer tự dùng `--resume last.pt`; chạy đến budget đã định.
8. Evaluate và verified backup cuối. Nếu run đã đủ max_steps, runner chỉ backup,
   bạn có thể đánh giá lại mà không train thêm.

Khác GPU/driver/kernel có thể làm số học khác; không hứa bitwise giữa phần cứng.
Nếu package versions khác snapshot, phục hồi đúng môi trường trước. Nếu Python
khác phiên bản, kiểm tra tương thích RNG/checkpoint và ghi rõ thay đổi; kiểm tra
resume vài bước trước run dài. GPU VRAM đủ không chứng minh forward B5 hợp lệ.

## Đầu ra cần giữ và cách gọi kết quả

```text
RUN_DIR/
  last.pt                         full checkpoint để resume
  config.json, run_config.yaml     config resolved và config launcher
  environment.json                Python/packages/CUDA/GPU/Git
  encoder_provenance.json          pretrained URI/revision/hash
  data.json                       số ảnh và hash manifests
  dataset_provenance.json          ZIP paths/size/SHA256
  requirements.runtime.txt
  run_notes.json
  metrics.jsonl, metrics_resume_*.jsonl
  eval_history.csv, eval_history.jsonl
  loss_curve_all.svg
  summary.json
  eval/step_<n>/{source_dev,target_val}/...
  standalone-target-val/{metrics.json,per_class_iou.csv,...png}
```

Gọi kết quả là **B0 source-only MiT-B5 trên GTA subset phần 01–03, fixed N
optimizer steps, Cityscapes val 500 ảnh**. Báo chính xác N, split, crop, batch,
eval_hw, timing/VRAM và selection protocol. Mức 1.000 steps không được xem là
ngưỡng chất lượng chuẩn. Overfit 2–4 ảnh thật và resume tests là kiểm chứng B0
riêng trong kế hoạch; smoke 20 steps không thay chúng.

## Kiểm chứng tài liệu và giới hạn

- Các CLI/config và checkpoint fields đối chiếu với commit pin ở cell 1.
- Cả 24 cell Python đã đạt kiểm tra cú pháp. Download, giải nén chạy lại,
  chọn đúng split, từ chối file sai CRC/hash và snapshot roundtrip đạt với dữ
  liệu nhỏ và Drive giả lập local.
- Chính cell runner đã được thử với trainer CPU tiny thật: train 0→2, lưu và
  xác minh snapshot, resume 2→4 rồi xác minh snapshot tiếp. Đây là kiểm chứng
  orchestration, không phải kết quả train MiT-B5 trên MoLab.
- Không khẳng định các cell đã chạy trên tài khoản MoLab/Google Drive của bạn.
  OAuth, quyền upload, quota, GPU và throughput cần kiểm chứng ở phiên thực tế.
- Source policy B0: [kế hoạch](CRISP_DA_IMPLEMENTATION_PLAN.md), §4 và §15.3–15.4.
- [MoLab Drive connection](https://marimo.io/blog/drive-storage).
- [MoLab persistence](https://marimo.io/pages/molab/storage).
- [gdrive-fsspec implementation](https://github.com/marimo-team/gdrive-fsspec):
  dùng public `info/get/put/makedirs`, không truy cập token hoặc private client.

Snapshot uploader nằm trong notebook theo tài liệu này; core trainer trong repo
vẫn không tự upload Drive nếu bạn gọi lệnh train dài trực tiếp bên ngoài runner.
