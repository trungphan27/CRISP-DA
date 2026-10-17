"""Bounded local ZIP extraction. Authentication/download is separate."""
import fnmatch
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from crisp_da.utils import sha256, write_json


def extract_zip(archive, destination, includes=("*",), max_uncompressed_gib=20):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Choose a new extraction directory; existing data is never overwritten")
    if max_uncompressed_gib <= 0:
        raise ValueError("Set a positive extraction limit")
    with zipfile.ZipFile(archive) as z:
        members, seen = [], set()
        for info in z.infolist():
            if not any(fnmatch.fnmatch(info.filename, pattern) for pattern in includes):
                continue
            path = PurePosixPath(info.filename)
            if (path.is_absolute() or ".." in path.parts or "\\" in info.filename
                    or ":" in info.filename or stat.S_ISLNK(info.external_attr >> 16)):
                raise ValueError(f"Unsafe ZIP member: {info.filename}")
            if not info.is_dir():
                key = path.as_posix().casefold()
                if key in seen:
                    raise ValueError(f"Duplicate ZIP member: {info.filename}")
                seen.add(key)
                members.append(info)
        if not members:
            raise ValueError("ZIP selection is empty")
        total = sum(info.file_size for info in members)
        if total > max_uncompressed_gib*1024**3:
            raise ValueError(f"Selection needs {total/1024**3:.2f} GiB, exceeding the explicit extraction limit")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="extract-",dir=destination.parent) as temp:
            stage = Path(temp)/"data"
            stage.mkdir()
            for info in members:
                output = stage/Path(*PurePosixPath(info.filename).parts)
                output.parent.mkdir(parents=True,exist_ok=True)
                with z.open(info) as src, output.open("xb") as dst:
                    # Reading to EOF validates this member's CRC.
                    shutil.copyfileobj(src,dst,1024*1024)
            report = {"archive_sha256":sha256(archive), "selected_members":len(members),
                      "uncompressed_bytes":total, "includes":list(includes),
                      "validation":"selected members read to EOF/CRC; unselected members not tested"}
            write_json(stage/"extraction_complete.json",report)
            stage.rename(destination)
    return report
