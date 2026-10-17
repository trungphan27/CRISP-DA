"""Open with `marimo edit notebooks/molab_train.py` after installing the project.

Repository clone, package installation and Drive OAuth are explicit bootstrap steps.
This launcher expects already extracted/verified data. It never downloads all ZIPs.
"""
import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # CRISP-DA — B0 source-only

    Cài package của repository vào Python notebook trước khi dùng launcher.
    Kết nối Drive bằng Files → Add remote storage. Tải và giải nén dữ liệu theo
    README_B0.md, sau đó tạo manifest. Smoke 20 bước không phải benchmark B0.
    Mỗi lần bấm nút chạy một tiến trình; chờ tiến trình kết thúc trước khi chạy tiếp.
    Checkpoint trong RUN_DIR cần được đồng bộ riêng về Drive trong các run dài.
    """)
    return


@app.cell
def _(mo):
    import os as _os
    from pathlib import Path as _Path
    project_input = mo.ui.text(value=str(_Path.cwd()), label="Repository root")
    data_input = mo.ui.text(value=_os.environ.get("DATA_ROOT", "/marimo/crisp-work/data"), label="DATA_ROOT")
    cache_input = mo.ui.text(value=_os.environ.get("CACHE_DIR", "/marimo/crisp-work/cache"), label="CACHE_DIR")
    run_input = mo.ui.text(value=_os.environ.get("RUN_DIR", "/marimo/crisp-work/runs/b0-smoke"), label="RUN_DIR")
    config_input = mo.ui.text(value="configs/b0_smoke.yaml", label="Config (relative to repository)")
    checkpoint_input = mo.ui.text(value="", label="Checkpoint path (evaluate/resume)")
    action_input = mo.ui.dropdown(options=["environment", "lock-encoder", "make-manifests", "train", "resume", "evaluate"],
                                  value="environment", label="Action")
    run_button = mo.ui.run_button(label="Chạy action đã chọn")
    mo.vstack([project_input,data_input,cache_input,run_input,config_input,checkpoint_input,action_input,run_button])
    return (project_input,data_input,cache_input,run_input,config_input,checkpoint_input,action_input,run_button)


@app.cell
def _(mo, project_input,data_input,cache_input,run_input,config_input,checkpoint_input,action_input,run_button):
    mo.stop(not run_button.value, mo.md("Chọn action rồi bấm nút để chạy."))
    import os as _os
    import subprocess as _subprocess
    import sys as _sys
    from pathlib import Path as _Path

    _root = _Path(project_input.value).expanduser().resolve()
    if not (_root/"pyproject.toml").is_file():
        raise FileNotFoundError("Repository root không chứa pyproject.toml")
    _env = _os.environ.copy()
    _env.update(DATA_ROOT=data_input.value,CACHE_DIR=cache_input.value,RUN_DIR=run_input.value)
    _action = action_input.value
    if _action == "environment":
        import shutil as _shutil
        from crisp_da.utils import environment as _environment
        print(_environment())
        print("Filesystem report (may not reflect platform quota):",_shutil.disk_usage(_root))
        _args = None
    elif _action == "lock-encoder":
        _args = ["lock-encoder","--cache-dir",cache_input.value,"--output","configs/mit_b5_imagenet.lock.json"]
    elif _action == "make-manifests":
        # Change encoding here only after inspecting your annotations.
        _args = ["make-manifests","--data-root",data_input.value,"--gta-encoding","raw_ids","--city-encoding","raw_ids"]
    elif _action == "evaluate":
        if not checkpoint_input.value:
            raise ValueError("Điền checkpoint path")
        _args = ["evaluate","--checkpoint",checkpoint_input.value,"--data-root",data_input.value,
                 "--output",str(_Path(run_input.value)/"standalone_eval")]
    else:
        _args = ["train","--config",config_input.value]
        if _action == "resume":
            if not checkpoint_input.value:
                raise ValueError("Điền checkpoint path")
            _args += ["--resume",checkpoint_input.value]
    if _args is not None:
        _subprocess.run([_sys.executable,"-m","crisp_da.cli",*_args],cwd=_root,env=_env,check=True)
    return


if __name__ == "__main__":
    app.run()
