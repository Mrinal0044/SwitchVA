"""Unit tests for Phase 2: Switch-Aware Encoding & SP-GSA."""

import pytest
import torch

from src.data.tokenizer_alignment import SubwordAligner, SUBWORD_LANG_MAP
from src.models.signed_switch_distance import SignedSwitchDistance
from src.models.switch_embedding import SwitchEmbedding
from src.models.multilingual_backbone import MultilingualBackbone
from src.models.sp_gsa import SPGSA
from src.models.switch_encoder import SwitchAwareEncoder, inspect_switch_gates


# TEST 1: Language ID preservation across alignment
def test_language_id_preservation():
    aligner = SubwordAligner(tokenizer=None)
    words = ["camera", "amazing", "hai"]
    langs = ["EN", "EN", "HI"]
    res = aligner.align_single(words, langs)

    assert "subword_lang_ids" in res
    assert "subword_lang_ids_numeric" in res
    # First and last are SPECIAL
    assert res["subword_lang_ids"][0] == "SPECIAL"
    assert res["subword_lang_ids"][-1] == "SPECIAL"
    # Content tokens match original
    assert res["subword_lang_ids"][1] == "EN"
    assert res["subword_lang_ids"][2] == "EN"
    assert res["subword_lang_ids"][3] == "HI"


# TEST 2: Word-to-subword alignment
def test_word_to_subword_alignment():
    aligner = SubwordAligner(tokenizer=None)
    words = ["word1", "word2"]
    langs = ["HI", "EN"]
    res = aligner.align_single(words, langs)

    assert len(res["word_to_subword"]) == 2
    # word 0 maps to index 1, word 1 maps to index 2
    assert res["word_to_subword"][0] == [1]
    assert res["word_to_subword"][1] == [2]


# TEST 3: Subword-to-word alignment
def test_subword_to_word_alignment():
    aligner = SubwordAligner(tokenizer=None)
    words = ["word1", "word2"]
    langs = ["HI", "EN"]
    res = aligner.align_single(words, langs)

    s_to_w = res["subword_to_word"]
    assert s_to_w[0] == -1  # [CLS]
    assert s_to_w[1] == 0   # word1
    assert s_to_w[2] == 1   # word2
    assert s_to_w[3] == -1  # [SEP]


# TEST 4: Signed switch distance with one switch
def test_signed_switch_distance_single_switch():
    calc = SignedSwitchDistance(max_distance=10)
    # [SPECIAL, HI, HI, EN, EN, SPECIAL]
    # Indices:    0,  1,  2,  3,  4,  5
    # Valid tokens: 1 (HI), 2 (HI), 3 (EN), 4 (EN)
    # Single switch point at index 3 (HI -> EN)
    lang_ids = torch.tensor([[3, 0, 0, 1, 1, 3]])
    raw_dists, dist_indices = calc(lang_ids)

    # Token 1 (HI): 1 - 3 = -2
    assert raw_dists[0, 1].item() == -2.0
    # Token 2 (HI): 2 - 3 = -1
    assert raw_dists[0, 2].item() == -1.0
    # Token 3 (EN): 3 - 3 = 0
    assert raw_dists[0, 3].item() == 0.0
    # Token 4 (EN): 4 - 3 = +1
    assert raw_dists[0, 4].item() == 1.0


# TEST 5: Signed switch distance with multiple switches
def test_signed_switch_distance_multiple_switches():
    calc = SignedSwitchDistance(max_distance=10)
    # [SPECIAL, HI, EN, HI, SPECIAL]
    # Indices:    0,  1,  2,  3,  4
    # Switch points at index 2 (1->2: HI->EN) and index 3 (2->3: EN->HI)
    lang_ids = torch.tensor([[3, 0, 1, 0, 3]])
    raw_dists, dist_indices = calc(lang_ids)

    # Token 1 (HI): nearest switch is 2 -> 1 - 2 = -1
    assert raw_dists[0, 1].item() == -1.0
    # Token 2 (EN): switch point -> 0
    assert raw_dists[0, 2].item() == 0.0
    # Token 3 (HI): switch point -> 0
    assert raw_dists[0, 3].item() == 0.0


# TEST 6: No-switch sequence (monolingual)
def test_signed_switch_distance_no_switch():
    calc = SignedSwitchDistance(max_distance=10)
    # [SPECIAL, HI, HI, HI, SPECIAL]
    lang_ids = torch.tensor([[3, 0, 0, 0, 3]])
    raw_dists, dist_indices = calc(lang_ids)

    # All tokens should have NO_SWITCH index (0)
    assert (dist_indices == calc.no_switch_idx).all()
    assert (raw_dists == 0.0).all()


# TEST 7: Switch at sequence boundary
def test_switch_at_sequence_boundary():
    calc = SignedSwitchDistance(max_distance=10)
    # First valid token is HI, second is EN -> switch at index 2
    lang_ids = torch.tensor([[3, 0, 1, 1, 3]])
    raw_dists, dist_indices = calc(lang_ids)
    assert raw_dists[0, 2].item() == 0.0


# TEST 8: Distance clipping
def test_distance_clipping():
    calc = SignedSwitchDistance(max_distance=2)
    # 5 HI tokens followed by EN at index 6
    lang_ids = torch.tensor([[3, 0, 0, 0, 0, 0, 1, 3]])
    raw_dists, dist_indices = calc(lang_ids)

    # Distance at index 1 is 1 - 6 = -5, clipped to -2
    assert raw_dists[0, 1].item() == -2.0


# TEST 9: Switch embedding forward & shapes
def test_switch_embedding_forward_and_shapes():
    hidden_dim = 64
    sw_embed = SwitchEmbedding(
        num_languages=5,
        lang_embed_dim=32,
        max_signed_distance=16,
        dist_embed_dim=32,
        output_dim=hidden_dim,
    )
    lang_ids = torch.tensor([[3, 0, 1, 0, 3], [3, 1, 1, 1, 3]])
    out = sw_embed(lang_ids)
    assert out.shape == (2, 5, hidden_dim)
    assert not torch.isnan(out).any()


# TEST 10: Padding behavior in switch embedding
def test_padding_behavior():
    sw_embed = SwitchEmbedding(output_dim=64, pad_lang_id=4)
    lang_ids = torch.tensor([[3, 0, 1, 4, 4]])  # 2 padding tokens at end
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]])

    out = sw_embed(lang_ids, attention_mask=attention_mask)
    # Padding positions must be zeroed out
    assert (out[0, 3:] == 0.0).all()


# TEST 11: Transformer wrapper forward
def test_transformer_wrapper():
    backbone = MultilingualBackbone(hidden_dim=64, use_mock=True)
    input_ids = torch.tensor([[10, 20, 30, 1], [40, 50, 1, 1]])
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

    H = backbone(input_ids, attention_mask=attention_mask)
    assert H.shape == (2, 4, 64)
    assert not torch.isnan(H).any()


# TEST 12: SP-GSA forward pass
def test_sp_gsa_forward_pass():
    hidden_dim = 64
    sp_gsa = SPGSA(hidden_dim=hidden_dim)
    H = torch.randn(2, 6, hidden_dim)
    E_switch = torch.randn(2, 6, hidden_dim)

    H_gated, G = sp_gsa(H, E_switch)
    assert H_gated.shape == (2, 6, hidden_dim)
    assert G.shape == (2, 6, hidden_dim)
    # Gating values must be strictly in [0, 1]
    assert (G >= 0.0).all() and (G <= 1.0).all()


# TEST 13: SP-GSA tensor dimensions
def test_sp_gsa_dimensions():
    hidden_dim = 128
    sp_gsa = SPGSA(hidden_dim=hidden_dim)
    H = torch.randn(4, 16, hidden_dim)
    E_switch = torch.randn(4, 16, hidden_dim)

    H_gated, G = sp_gsa(H, E_switch)
    assert H_gated.size(0) == 4
    assert H_gated.size(1) == 16
    assert H_gated.size(2) == 128


# TEST 14: Gradient propagation through SP-GSA
def test_gradient_propagation():
    hidden_dim = 32
    encoder = SwitchAwareEncoder(hidden_dim=hidden_dim, use_mock_backbone=True)

    input_ids = torch.tensor([[5, 12, 18, 25, 2]])
    lang_ids = torch.tensor([[3, 0, 1, 0, 3]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1]])

    outputs = encoder(input_ids, lang_ids, attention_mask=attention_mask)
    target = torch.randn_like(outputs["H_gated"])
    loss = (outputs["H_gated"] * target).sum()
    loss.backward()

    # Check that backbone and gating parameters received gradients
    assert encoder.sp_gsa.gate_proj.weight.grad is not None
    assert encoder.sp_gsa.gate_proj.weight.grad.abs().sum() > 0


# TEST 15: NaN/Inf detection
def test_nan_inf_detection():
    encoder = SwitchAwareEncoder(hidden_dim=32, use_mock_backbone=True)
    input_ids = torch.randint(0, 1000, (3, 10))
    lang_ids = torch.randint(0, 3, (3, 10))

    outputs = encoder(input_ids, lang_ids)
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            assert not torch.isnan(v).any(), f"NaN detected in {k}"
            assert not torch.isinf(v).any(), f"Inf detected in {k}"


# TEST 16: Batch with variable sequence lengths & masking
def test_variable_sequence_lengths_masking():
    encoder = SwitchAwareEncoder(hidden_dim=32, use_mock_backbone=True)
    input_ids = torch.tensor([[10, 20, 30, 40], [50, 60, 1, 1]])  # second has length 2 + pad 2
    lang_ids = torch.tensor([[3, 0, 1, 3], [3, 0, 4, 4]])
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])

    outputs = encoder(input_ids, lang_ids, attention_mask=attention_mask)
    # Check that padding positions in H_gated are zeroed
    assert (outputs["H_gated"][1, 2:] == 0.0).all()
    assert (outputs["gate_values"][1, 2:] == 0.0).all()
