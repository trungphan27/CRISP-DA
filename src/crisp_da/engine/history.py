"""One durable evaluation table across training chunks and resume sessions."""
import csv
import io
import json
import os
from pathlib import Path

from crisp_da.data.mapping import CLASS_NAMES


def _write_history(run_dir, records):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    records = sorted(records, key=lambda r: (r["step"], r["completed_epochs"], r["source_position"], r["split"]))
    fields = ["epoch", "completed_epochs", "source_position", "step", "split", "trigger",
              "loss_ce", "miou", "pixel_accuracy", "samples", "valid_pixels", "classes_with_union"]
    fields += ["iou_" + name.replace(" ", "_") for name in CLASS_NAMES]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for record in records:
        row = {key: record.get(key) for key in fields if not key.startswith("iou_")}
        row["trigger"] = "+".join(record["triggers"])
        row.update({"iou_" + name.replace(" ", "_"): record["per_class_iou"][name] for name in CLASS_NAMES})
        writer.writerow(row)
    # JSONL is the canonical history; CSV is a convenient plotting export.
    for name, text in (
        ("eval_history.jsonl", "".join(json.dumps(r, allow_nan=False) + "\n" for r in records)),
        ("eval_history.csv", stream.getvalue()),
    ):
        path = run_dir / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)


def read_eval_history(run_dir):
    path = Path(run_dir) / "eval_history.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def save_eval_record(run_dir, record):
    records = read_eval_history(run_dir)
    key = lambda r: (r["step"], r["completed_epochs"], r["source_position"], r["split"])
    records = [r for r in records if key(r) != key(record)]
    _write_history(run_dir, [*records, record])


def align_eval_history(run_dir, step=None, completed_epochs=0, source_position=0):
    """Discard stale future records when restoring an older checkpoint locally."""
    records = read_eval_history(run_dir) if step is not None else []
    records = [r for r in records if r["step"] <= step
               and (r["completed_epochs"], r["source_position"]) <= (completed_epochs, source_position)]
    _write_history(run_dir, records)
