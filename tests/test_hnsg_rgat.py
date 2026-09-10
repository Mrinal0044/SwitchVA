"""Tests for H-NSG construction and RGAT message passing."""

import torch
import pytest
from src.models.hnsg import HNSGBuilder, NUM_RELATIONS
from src.models.rgat import RGAT, RGATLayer


def test_hnsg_builder():
    batch_size = 2
    seq_len = 8
    dim = 64

    builder = HNSGBuilder(node_dim=dim, nrc_vad_dim=3)

    tokens = torch.randn(batch_size, seq_len, dim)
    aspect_vec = torch.randn(batch_size, dim)
    opinion_vec = torch.randn(batch_size, dim)
    vad_priors = torch.tensor([[0.7, 0.6, 0.5], [0.2, 0.8, 0.4]])

    node_feats, adj = builder(
        token_states=tokens,
        aspect_repr=aspect_vec,
        opinion_repr=opinion_vec,
        nrc_vad_priors=vad_priors,
    )

    # Total nodes = seq_len + 4 (aspect, opinion, switch, vad)
    assert node_feats.shape == (batch_size, seq_len + 4, dim)
    assert adj.shape == (batch_size, NUM_RELATIONS, seq_len + 4, seq_len + 4)


def test_rgat_forward():
    batch_size = 2
    num_nodes = 12
    dim = 64
    seq_len = 8

    rgat = RGAT(
        hidden_dim=dim,
        num_layers=2,
        num_heads=4,
        num_relations=NUM_RELATIONS,
        dropout=0.1,
    )

    node_feats = torch.randn(batch_size, num_nodes, dim)
    adj = torch.zeros(batch_size, NUM_RELATIONS, num_nodes, num_nodes)
    for i in range(num_nodes):
        adj[:, :, i, i] = 1.0  # self loops

    all_nodes, graph_aspect = rgat(node_feats, adj, seq_len=seq_len)

    assert all_nodes.shape == (batch_size, num_nodes, dim)
    assert graph_aspect.shape == (batch_size, dim)
    assert not torch.isnan(all_nodes).any()
