"""Branch 2: Heterogeneous Neuro-Symbolic Graph (H-NSG) representation and builder."""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from src.data.nrc_vad import NRCVADLexicon
from src.data.dependency_parser import DependencyParserInterface, HeuristicDependencyBuilder

# Three explicit, distinguishable relation types in H-NSG
REL_SYNTACTIC = 0
REL_SWITCH = 1
REL_ASPECT_OPINION = 2
NUM_RELATIONS = 3

RELATION_NAMES = {
    REL_SYNTACTIC: "SYNTACTIC",
    REL_SWITCH: "SWITCH",
    REL_ASPECT_OPINION: "ASPECT_OPINION",
}


class HNSGOutput(tuple):
    """Container supporting both tuple unpacking and dictionary key access."""
    def __new__(cls, node_features, nrc_vad_priors, nrc_vad_mask=None, relational_adj=None, node_info=None):
        if relational_adj is not None and (nrc_vad_mask is None or isinstance(nrc_vad_priors, torch.Tensor) and nrc_vad_mask.dim() == 4):
            # If unpacked as (node_feats, adj)
            return super().__new__(cls, (node_features, nrc_vad_priors))
        return super().__new__(cls, (node_features, nrc_vad_priors, nrc_vad_mask))

    def __init__(self, node_features, nrc_vad_priors, nrc_vad_mask=None, relational_adj=None, node_info=None):
        self.node_features = node_features
        self.nrc_vad_priors = nrc_vad_priors
        self.nrc_vad_mask = nrc_vad_mask
        self.relational_adj = relational_adj if relational_adj is not None else nrc_vad_priors
        self.node_info = node_info or {}

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in ("node_features", "H_init"):
                return self.node_features
            elif item in ("nrc_vad_priors", "vad_priors"):
                return self.nrc_vad_priors
            elif item in ("nrc_vad_mask", "vad_mask"):
                return self.nrc_vad_mask
            elif item in ("relational_adj", "adj_matrix", "adj"):
                return self.relational_adj
            elif item in self.node_info:
                return self.node_info[item]
            raise KeyError(f"Key {item} not found in HNSGOutput")
        return super().__getitem__(item)


class HeterogeneousNeuroSymbolicGraph(nn.Module):
    """Heterogeneous Neuro-Symbolic Graph (H-NSG) Builder (Branch 2).

    Constructs graph node representations and a 3D relational adjacency tensor:
        (batch_size, num_relations, seq_len, seq_len)

    Relations:
    0: SYNTACTIC       - Syntactic dependency edges between governors and dependents
    1: SWITCH          - Code-switch transition bridges between language boundaries
    2: ASPECT_OPINION  - Semantic links between aspect spans and opinion spans

    Symbolic Priors:
    - Integrates NRC-VAD affective norms [Valence, Arousal] with availability masking.
    - Ground-truth dataset VA targets are NEVER used as inputs (target-leakage safeguard).
    """

    def __init__(
        self,
        node_dim: int = 768,
        d_model: Optional[int] = None,
        vad_dim: int = 2,
        nrc_vad_dim: Optional[int] = None,
        num_relations: int = NUM_RELATIONS,
        lexicon: Optional[NRCVADLexicon] = None,
        dependency_parser: Optional[DependencyParserInterface] = None,
    ):
        """
        Args:
            node_dim: Dimension d of node representations (matches H_gated).
            d_model: Alias for node_dim.
            vad_dim: Dimension of NRC-VAD prior [valence, arousal] (default 2).
            nrc_vad_dim: Alias for vad_dim.
            num_relations: Number of distinct relation types (default 3).
            lexicon: Optional custom NRCVADLexicon instance.
            dependency_parser: Optional pluggable dependency parser instance.
        """
        super().__init__()
        dim = d_model or node_dim
        v_dim = nrc_vad_dim or vad_dim
        self.node_dim = dim
        self.d_model = dim
        self.vad_dim = v_dim
        self.num_relations = num_relations
        self.dep_parser = dependency_parser or HeuristicDependencyBuilder()
        self.lexicon = lexicon or NRCVADLexicon()

        # Learnable projection for symbolic NRC-VAD priors: R^v_dim -> R^d
        self.prior_proj = nn.Sequential(
            nn.Linear(v_dim, dim // 4 if dim >= 8 else dim),
            nn.GELU(),
            nn.Linear(dim // 4 if dim >= 8 else dim, dim),
            nn.LayerNorm(dim),
        )
        self.vad_proj = self.prior_proj  # Alias

        # Special node projections for legacy compatibility
        self.aspect_proj = nn.Linear(dim, dim)
        self.opinion_proj = nn.Linear(dim, dim)
        self.switch_marker_proj = nn.Linear(dim, dim)

    def extract_lexicon_priors(
        self,
        tokens_batch: List[List[str]],
        max_seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract NRC-VAD priors for batch of token lists."""
        batch_size = len(tokens_batch)
        if max_seq_len is None:
            max_seq_len = max((len(t) for t in tokens_batch), default=1)

        priors_list: List[List[List[float]]] = []
        mask_list: List[List[bool]] = []

        for b in range(batch_size):
            toks = tokens_batch[b]
            cur_toks = toks[:max_seq_len] + ["<pad>"] * max(0, max_seq_len - len(toks))
            p_b, m_b = self.lexicon.get_sentence_priors(cur_toks[:max_seq_len])
            priors_list.append(p_b)
            mask_list.append(m_b)

        dev = device or torch.device("cpu")
        vad_priors = torch.tensor(priors_list, dtype=torch.float32, device=dev)
        vad_mask = torch.tensor(mask_list, dtype=torch.bool, device=dev)
        return vad_priors, vad_mask

    def build_graph(
        self,
        tokens_batch: List[List[str]],
        lang_ids_batch: Optional[Union[List[List[int]], List[List[str]]]] = None,
        aspect_opinion_pairs_batch: Optional[List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]] = None,
        aspect_spans_batch: Optional[List[List[Dict[str, Any]]]] = None,
        opinion_spans_batch: Optional[List[List[Dict[str, Any]]]] = None,
        max_seq_len: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_node_info: bool = False,
        device: Optional[torch.device] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Dict[str, Any]]]]:
        """Construct multi-relational adjacency tensor (B, 3, N, N)."""
        batch_size = len(tokens_batch)
        if max_seq_len is None:
            max_seq_len = max((len(t) for t in tokens_batch), default=1)

        if device is None and attention_mask is not None:
            device = attention_mask.device
        elif device is None:
            device = torch.device("cpu")

        adj = torch.zeros(
            (batch_size, self.num_relations, max_seq_len, max_seq_len),
            dtype=torch.float32,
            device=device,
        )

        node_info: List[Dict[str, Any]] = []

        for b in range(batch_size):
            tokens = tokens_batch[b] if b < len(tokens_batch) else []
            langs = lang_ids_batch[b] if (lang_ids_batch is not None and b < len(lang_ids_batch)) else []
            n_tokens = min(len(tokens), max_seq_len)

            if attention_mask is not None:
                n_tokens = min(n_tokens, int(attention_mask[b].sum().item()))

            t_to_g = {i: i for i in range(n_tokens)}
            g_to_t = {i: i for i in range(n_tokens)}
            node_info.append({"token_to_node": t_to_g, "node_to_token": g_to_t})

            if n_tokens <= 0:
                continue

            # -------------------------------------------------------------
            # Relation 0: SYNTACTIC DEPENDENCY EDGES + SELF-LOOPS
            # -------------------------------------------------------------
            for i in range(n_tokens):
                adj[b, REL_SYNTACTIC, i, i] = 1.0

            dep_edges = self.dep_parser.parse(tokens[:n_tokens])
            for head_idx, dep_idx, _ in dep_edges:
                if head_idx < n_tokens and dep_idx < n_tokens:
                    adj[b, REL_SYNTACTIC, head_idx, dep_idx] = 1.0

            # -------------------------------------------------------------
            # Relation 1: CODE-SWITCH TRANSITION BRIDGES
            # -------------------------------------------------------------
            for i in range(1, n_tokens):
                l_prev = langs[i - 1] if i - 1 < len(langs) else None
                l_curr = langs[i] if i < len(langs) else None

                is_switch = False
                if isinstance(l_prev, str) and isinstance(l_curr, str):
                    lp, lc = l_prev.upper(), l_curr.upper()
                    if lp != lc and lp not in ["SPECIAL", "PAD"] and lc not in ["SPECIAL", "PAD"]:
                        is_switch = True
                elif isinstance(l_prev, (int, float)) and isinstance(l_curr, (int, float)):
                    if l_prev != l_curr and l_prev in [0, 1, 2] and l_curr in [0, 1, 2]:
                        is_switch = True

                if is_switch:
                    adj[b, REL_SWITCH, i - 1, i] = 1.0
                    adj[b, REL_SWITCH, i, i - 1] = 1.0

            # -------------------------------------------------------------
            # Relation 2: ASPECT-OPINION SEMANTIC LINKS
            # -------------------------------------------------------------
            if aspect_opinion_pairs_batch is not None and b < len(aspect_opinion_pairs_batch):
                for (asp_s, asp_e), (op_s, op_e) in aspect_opinion_pairs_batch[b]:
                    if 0 <= asp_s <= asp_e < n_tokens and 0 <= op_s <= op_e < n_tokens:
                        for i_tok in range(asp_s, asp_e + 1):
                            for j_tok in range(op_s, op_e + 1):
                                adj[b, REL_ASPECT_OPINION, i_tok, j_tok] = 1.0
                                adj[b, REL_ASPECT_OPINION, j_tok, i_tok] = 1.0

            if aspect_spans_batch is not None and opinion_spans_batch is not None:
                asp_list = aspect_spans_batch[b] if b < len(aspect_spans_batch) else []
                op_list = opinion_spans_batch[b] if b < len(opinion_spans_batch) else []
                min_pairs = min(len(asp_list), len(op_list))
                for k in range(min_pairs):
                    asp_s = asp_list[k].get("start", -1)
                    asp_e = asp_list[k].get("end", -1)
                    op_s = op_list[k].get("start", -1)
                    op_e = op_list[k].get("end", -1)
                    if 0 <= asp_s <= asp_e < n_tokens and 0 <= op_s <= op_e < n_tokens:
                        for i_tok in range(asp_s, asp_e + 1):
                            for j_tok in range(op_s, op_e + 1):
                                adj[b, REL_ASPECT_OPINION, i_tok, j_tok] = 1.0
                                adj[b, REL_ASPECT_OPINION, j_tok, i_tok] = 1.0

        if return_node_info:
            return adj, node_info
        return adj

    def build_relational_adjacency(
        self,
        batch_size: int,
        seq_len: int,
        tokens_batch: List[List[str]],
        lang_ids_batch: Optional[Union[List[List[int]], List[List[str]]]] = None,
        aspect_spans_batch: Optional[List[List[Dict[str, Any]]]] = None,
        opinion_spans_batch: Optional[List[List[Dict[str, Any]]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Alias for build_graph adhering to explicit signature."""
        return self.build_graph(
            tokens_batch=tokens_batch,
            lang_ids_batch=lang_ids_batch,
            aspect_spans_batch=aspect_spans_batch,
            opinion_spans_batch=opinion_spans_batch,
            max_seq_len=seq_len,
            attention_mask=attention_mask,
            device=device,
        )

    def forward(
        self,
        H_gated: Optional[torch.Tensor] = None,
        tokens_batch: Optional[List[List[str]]] = None,
        lang_ids_batch: Optional[Union[List[List[int]], List[List[str]]]] = None,
        aspect_spans_batch: Optional[List[List[Dict[str, Any]]]] = None,
        opinion_spans_batch: Optional[List[List[Dict[str, Any]]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        nrc_vad_priors: Optional[torch.Tensor] = None,
        nrc_vad_mask: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
        # Legacy positional/keyword arguments
        token_states: Optional[torch.Tensor] = None,
        aspect_repr: Optional[torch.Tensor] = None,
        opinion_repr: Optional[torch.Tensor] = None,
        adj_matrices: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        """Construct graph node representations with NRC-VAD priors and relational edges."""
        # Handle legacy input signature if token_states is provided
        if token_states is not None:
            H_gated = token_states
            batch_size, seq_len, dim = H_gated.shape
            device = H_gated.device

            # If aspect_repr and opinion_repr provided, append 4 virtual nodes
            extra_nodes = []
            if aspect_repr is not None:
                extra_nodes.append(self.aspect_proj(aspect_repr).unsqueeze(1))
            if opinion_repr is not None:
                extra_nodes.append(self.opinion_proj(opinion_repr).unsqueeze(1))
            
            # Switch marker & VAD prior node
            switch_node = self.switch_marker_proj(H_gated.mean(dim=1, keepdim=True))
            extra_nodes.append(switch_node)
            
            if nrc_vad_priors is not None:
                vad_feat = self.prior_proj(nrc_vad_priors).unsqueeze(1)
                extra_nodes.append(vad_feat)
            else:
                extra_nodes.append(torch.zeros(batch_size, 1, dim, device=device))

            all_nodes = torch.cat([H_gated] + extra_nodes, dim=1)
            total_n = all_nodes.shape[1]
            adj = torch.zeros(batch_size, self.num_relations, total_n, total_n, device=device)
            for i in range(total_n):
                adj[:, :, i, i] = 1.0
            return all_nodes, adj

        assert H_gated is not None, "H_gated must be provided"
        batch_size, seq_len, dim = H_gated.shape
        device = H_gated.device
        eff_len = max_seq_len or seq_len

        # 1. Compute NRC-VAD Priors and Mask if not provided
        if nrc_vad_priors is None or nrc_vad_mask is None:
            if tokens_batch is not None:
                nrc_vad_priors, nrc_vad_mask = self.extract_lexicon_priors(
                    tokens_batch=tokens_batch,
                    max_seq_len=eff_len,
                    device=device,
                )
            else:
                nrc_vad_priors = torch.zeros(batch_size, eff_len, self.vad_dim, device=device)
                nrc_vad_mask = torch.zeros(batch_size, eff_len, dtype=torch.bool, device=device)

        # 2. Integrate Symbolic NRC-VAD Prior into Node Features
        vad_projected = self.prior_proj(nrc_vad_priors)
        gated_vad = vad_projected * nrc_vad_mask.unsqueeze(-1).float()
        node_features = H_gated + gated_vad

        if attention_mask is not None:
            node_features = node_features * attention_mask.unsqueeze(-1).float()

        # 3. Construct 3D Relational Adjacency Tensor
        relational_adj, node_info = self.build_graph(
            tokens_batch=tokens_batch or [[] for _ in range(batch_size)],
            lang_ids_batch=lang_ids_batch,
            aspect_spans_batch=aspect_spans_batch,
            opinion_spans_batch=opinion_spans_batch,
            max_seq_len=eff_len,
            attention_mask=attention_mask,
            return_node_info=True,
            device=device,
        )

        return HNSGOutput(
            node_features=node_features,
            nrc_vad_priors=nrc_vad_priors,
            nrc_vad_mask=nrc_vad_mask,
            relational_adj=relational_adj,
            node_info=node_info,
        )


# Backward-compatible alias
HNSGBuilder = HeterogeneousNeuroSymbolicGraph
