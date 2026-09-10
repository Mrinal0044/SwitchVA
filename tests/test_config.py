"""Tests for configuration loading and validation."""

import os
import pytest
from src.utils.config import load_config, Config
from src.utils.device import get_device
from src.utils.seed import set_seed


def test_load_nssg_dimnet_config():
    config_path = "configs/nssg_dimnet.yaml"
    assert os.path.exists(config_path), f"Config file not found: {config_path}"
    cfg = load_config(config_path)

    assert cfg.model.name == "NSSGDimNet"
    assert cfg.model.backbone.hidden_dim == 768
    assert cfg.model.switch_embedding.max_signed_distance == 64
    assert cfg.model.cross_attention.fused_dim == 2304
    assert cfg.loss.ccc_weight_v == 1.0


def test_load_baseline_config():
    config_path = "configs/baseline.yaml"
    assert os.path.exists(config_path)
    cfg = load_config(config_path)
    assert cfg.model.name in ["BaselineTransformer", "SwitchUnawareBaseline"]


def test_load_ablations_config():
    config_path = "configs/ablations.yaml"
    assert os.path.exists(config_path)
    cfg = load_config(config_path)
    assert "wo_switch_embedding" in cfg.ablation_studies
    assert "wo_sp_gsa" in cfg.ablation_studies
    assert "wo_hnsg_rgat" in cfg.ablation_studies


def test_device_selection():
    device = get_device("cpu")
    assert device.type == "cpu"

    device_auto = get_device("auto")
    assert device_auto.type in ["cuda", "mps", "cpu"]


def test_set_seed():
    set_seed(42)
