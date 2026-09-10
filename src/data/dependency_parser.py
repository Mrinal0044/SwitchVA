"""Pluggable dependency parser interface and dependency graph builder."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class DependencyParserInterface(ABC):
    """Abstract base interface for syntactic dependency parsing."""

    @abstractmethod
    def parse(self, tokens: List[str]) -> List[Tuple[int, int, str]]:
        """Parse a list of tokens into dependency relations.

        Args:
            tokens: List of token strings in the sequence.

        Returns:
            List of (head_index, dependent_index, dependency_relation_label).
        """
        pass


class HeuristicDependencyBuilder(DependencyParserInterface):
    """Deterministic, pluggable syntactic dependency builder for code-switched text.

    Constructs directed dependency relations connecting governor/head tokens
    and dependent modifiers, noun-chunks, and root predicates without requiring
    heavy external binary dependencies.
    """

    def __init__(self, window_size: int = 2):
        self.window_size = window_size

    def parse(self, tokens: List[str]) -> List[Tuple[int, int, str]]:
        """Construct directed syntactic dependency edges for token sequence.

        Returns:
            List of (head_idx, dep_idx, relation_type_str).
        """
        edges: List[Tuple[int, int, str]] = []
        n = len(tokens)
        if n <= 1:
            return edges

        # Identify pseudo-root (predicate / central token)
        root_idx = max(0, min(n - 1, n // 2))

        for i in range(n):
            # 1. Local syntactic adjacency & modifier dependencies
            for w in range(1, self.window_size + 1):
                if i + w < n:
                    edges.append((i, i + w, "advmod" if w == 1 else "nmod"))
                    # Explicit reverse edge marked with distinct reverse label for bidirectional propagation
                    edges.append((i + w, i, "rev_advmod" if w == 1 else "rev_nmod"))

            # 2. Clausal root attachment
            if i != root_idx and abs(i - root_idx) > self.window_size:
                edges.append((root_idx, i, "root_dep"))
                edges.append((i, root_idx, "rev_root_dep"))

        return edges
