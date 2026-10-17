import torch
import pytest

from crisp_da.config import load_config
from crisp_da.data.synthetic import create_synthetic


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def fixture_config(tmp_path):
    root = tmp_path/"data"
    create_synthetic(root)
    cfg = load_config("configs/b0_tiny_test.yaml")
    cfg["paths"]["data_root"] = str(root)
    cfg["paths"]["run_dir"] = str(tmp_path/"run")
    return cfg
