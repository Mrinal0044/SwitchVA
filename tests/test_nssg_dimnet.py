"""End-to-end integration tests for NSSG-DimNet model architecture."""

import torch
import pytest
from src.models.nssg_dimnet import NSSGDimNet
from src.utils.config import load_config


def test_nssg_dimnet_forward_pass():
    batch_size = 2
    seq_len = 16
    hidden_dim = 64

    model = NSSGDimNet(
        hidden_dim=hidden_dim,
        num_languages=4,
        max_signed_distance=32,
        num_spgsa_heads=4,
        span_mlp_dim=32,
        nrc_vad_dim=3,
        num_relations=6,
        rgat_layers=2,
        rgat_heads=4,
        cross_attn_heads=4,
        regression_hidden_dim=32,
        dropout=0.1,
        use_mock_backbone=True,
    )

    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    lang_ids = torch.randint(0, 3, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    aspect_spans = torch.tensor([[1, 3], [4, 6]])
    opinion_spans = torch.tensor([[4, 7], [7, 9]])
    nrc_vad_priors = torch.tensor([[0.6, 0.4, 0.5], [0.3, 0.7, 0.2]])

    outputs = model(
        input_ids=input_ids,
        lang_ids=lang_ids,
        attention_mask=attention_mask,
        aspect_spans=aspect_spans,
        opinion_spans=opinion_spans,
        nrc_vad_priors=nrc_vad_priors,
    )

    assert "valence" in outputs
    assert "arousal" in outputs
    assert "fused_representation" in outputs
    assert "span_logits" in outputs

    assert outputs["valence"].shape == (batch_size, 1)
    assert outputs["arousal"].shape == (batch_size, 1)
    assert outputs["fused_representation"].shape == (batch_size, 3 * hidden_dim)

    # Check bounds
    assert (outputs["valence"] >= 0.0).all() and (outputs["valence"] <= 1.0).all()
    assert (outputs["arousal"] >= 0.0).all() and (outputs["arousal"] <= 1.0).all()


def test_nssg_dimnet_from_config():
    cfg = load_config("configs/nssg_dimnet.yaml")
    # instantiate with mock backbone for quick offline test
    model = NSSGDimNet(config=cfg, use_mock_backbone=True)
    assert model.hidden_dim == cfg.model.backbone.hidden_dim
