import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from crisp_da.config import load_config
from crisp_da.engine.checkpoint import load_checkpoint
from crisp_da.engine.evaluate import confusion_update, metrics_from_confusion, evaluate
from crisp_da.engine.optim import WarmupPoly, build_optimizer, supervised_ce
from crisp_da.engine.train import train, make_datasets, val_loader
from crisp_da.models.segmentor import build_model
from crisp_da.utils import seed_all


def test_main_config_blocks_unset_benchmark_budget():
    with pytest.raises(ValueError,match="max_steps"):
        load_config("configs/b0_source_only.yaml")


def test_model_contract_and_ignore_gradient(fixture_config):
    model = build_model(fixture_config)
    model.eval()
    out = model(torch.randn(1,3,64,96))
    assert out["dense_features"].shape == (1,256,16,24)
    assert out["logits_low"].shape == (1,19,16,24)
    assert out["logits"].shape == (1,19,64,96)
    logits = torch.randn(1,19,2,2,requires_grad=True)
    labels = torch.tensor([[[0,255],[1,255]]])
    supervised_ce(logits,labels).backward()
    assert torch.count_nonzero(logits.grad[:,:,:,1]) == 0
    assert torch.count_nonzero(logits.grad[:,:,:,0]) > 0
    logits.grad = None
    loss = supervised_ce(logits,torch.full_like(labels,255))
    loss.backward()
    assert loss.item() == 0 and torch.count_nonzero(logits.grad) == 0


def test_global_confusion_and_absent_class_policy():
    matrix = torch.zeros(19,19,dtype=torch.int64)
    confusion_update(matrix,torch.tensor([0,1,1,18]),torch.tensor([0,0,1,255]))
    result = metrics_from_confusion(matrix)
    assert result["valid_pixels"] == 3
    assert result["miou"] == .5
    assert result["per_class_iou"]["bicycle"] is None
    assert result["classes_with_union"] == 2


def test_eval_interpolates_logits_before_argmax_at_original_gt_size():
    class Fixed(torch.nn.Module):
        def forward(self,image):
            logits = torch.full((1,19,2,2),-100.)
            logits[:,1] = 0.
            logits[:,0] = torch.tensor([[1.,-10.],[1.,1.]])
            return {"logits":logits}
    # Bilinear logits differ from resizing the hard 2x2 prediction.
    labels = torch.tensor([[[0,1,1],[0,1,1],[0,0,0]]])
    batch = {"image":torch.zeros(1,3,2,2),"label":labels,"sample_id":["fixture"]}
    result = evaluate(Fixed(),[batch],torch.device("cpu"))
    assert result["miou"] == 1.0 and result["valid_pixels"] == 9


def test_scheduler_update_and_optimizer_partition(fixture_config):
    model = build_model(fixture_config)
    opt = build_optimizer(model,fixture_config["train"])
    sched = WarmupPoly(opt,fixture_config["train"])
    ids = [id(p) for group in opt.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    initial = opt.param_groups[0]["lr"]
    for _ in range(fixture_config["train"]["max_steps"]):
        sched.step()
    assert opt.param_groups[0]["lr"] == 0 and initial > 0


def test_resume_matches_uninterrupted_training(fixture_config,tmp_path):
    cfg = fixture_config
    summary = train(cfg)
    assert summary["training_finished"]
    full = load_checkpoint(Path(cfg["paths"]["run_dir"])/"last.pt")
    resumed_cfg = copy.deepcopy(cfg)
    resumed_cfg["paths"]["run_dir"] = str(tmp_path/"resume")
    partial = train(resumed_cfg,stop_after=1)
    assert not partial["training_finished"]
    train(resumed_cfg,resume=Path(resumed_cfg["paths"]["run_dir"])/"last.pt")
    restored = load_checkpoint(Path(resumed_cfg["paths"]["run_dir"])/"last.pt")
    for key in full["model"]:
        torch.testing.assert_close(full["model"][key],restored["model"][key],atol=0,rtol=0)
    assert full["sampler"] == restored["sampler"]
    assert full["scheduler"] == restored["scheduler"]
    for key in full["optimizer"]["state"]:
        for field in full["optimizer"]["state"][key]:
            torch.testing.assert_close(full["optimizer"]["state"][key][field],restored["optimizer"]["state"][key][field],atol=0,rtol=0)
    model = build_model(cfg,False,full["encoder_config"])
    model.load_state_dict(full["model"])
    ds,_ = make_datasets(cfg)
    before = evaluate(model,val_loader(ds["target_val"]),torch.device("cpu"))
    model.load_state_dict(restored["model"])
    after = evaluate(model,val_loader(ds["target_val"]),torch.device("cpu"))
    assert before == after and before["samples"] == 2
    assert Path(cfg["paths"]["run_dir"],"loss_curve.svg").exists()


def test_resume_rejects_changed_manifest(fixture_config):
    train(fixture_config,stop_after=1)
    path = Path(fixture_config["paths"]["data_root"])/"manifests/source_train.jsonl"
    path.write_text(path.read_text()+"\n")
    with pytest.raises(ValueError,match="manifests changed"):
        train(fixture_config,resume=Path(fixture_config["paths"]["run_dir"])/"last.pt")


def test_tiny_model_can_overfit_fixed_source_batch(fixture_config):
    seed_all(7)
    model = build_model(fixture_config)
    cfg = dict(fixture_config["train"],head_lr=.003,backbone_lr=.0003)
    opt = build_optimizer(model,cfg)
    image, labels = torch.randn(2,3,32,32),torch.zeros(2,32,32,dtype=torch.long)
    losses = []
    for _ in range(12):
        opt.zero_grad()
        loss = supervised_ce(model(image)["logits"],labels)
        losses.append(loss.item())
        loss.backward()
        opt.step()
    assert losses[-1] < losses[0]*.5, losses


def test_all_ignore_batches_never_update_optimizer(fixture_config,monkeypatch):
    from crisp_da.data.gta5 import GTA5Dataset
    original = GTA5Dataset.__getitem__
    def ignored(self,key):
        sample = original(self,key)
        sample["label"].fill_(255)
        return sample
    monkeypatch.setattr(GTA5Dataset,"__getitem__",ignored)
    fixture_config["train"]["max_skipped_updates"] = 2
    with pytest.raises(RuntimeError,match="skipped"):
        train(fixture_config)
    logs = [json.loads(line) for line in Path(fixture_config["paths"]["run_dir"],"metrics.jsonl").read_text().splitlines()]
    assert all(row["event"] == "skip" and row["step"] == 0 for row in logs)
