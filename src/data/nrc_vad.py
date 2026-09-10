"""Symbolic NRC-VAD Affective Lexicon Norms and Prior Lookup module."""

import re
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Sample core lexicon covering English and common transliterated Hindi affective tokens
# Full NRC-VAD norms are bounded in [0, 1] for Valence, Arousal, Dominance
EMBEDDED_NRC_VAD: Dict[str, Tuple[float, float, float]] = {
    # Positive / High Valence
    "amazing": (0.950, 0.720, 0.760),
    "love": (0.980, 0.830, 0.820),
    "good": (0.850, 0.550, 0.650),
    "best": (0.940, 0.690, 0.780),
    "happy": (0.960, 0.730, 0.750),
    "great": (0.910, 0.660, 0.730),
    "nice": (0.820, 0.480, 0.610),
    "funny": (0.830, 0.610, 0.640),
    "free": (0.780, 0.510, 0.630),
    "download": (0.620, 0.450, 0.520),
    "win": (0.920, 0.780, 0.810),
    "winner": (0.930, 0.770, 0.820),
    "achhi": (0.840, 0.510, 0.630),
    "accha": (0.800, 0.500, 0.630),
    "wadia": (0.860, 0.550, 0.650),
    "pyaar": (0.950, 0.800, 0.800),
    "sundar": (0.880, 0.520, 0.670),
    "shandar": (0.930, 0.710, 0.790),
    "badiya": (0.870, 0.560, 0.660),
    "mubarak": (0.920, 0.680, 0.750),
    "thanks": (0.850, 0.470, 0.600),
    "shukriya": (0.860, 0.480, 0.610),
    "respect": (0.910, 0.580, 0.790),
    "battery": (0.500, 0.400, 0.500),
    "camera": (0.550, 0.450, 0.500),
    "phone": (0.500, 0.500, 0.500),

    # Negative / Low Valence
    "bad": (0.150, 0.650, 0.350),
    "worst": (0.050, 0.720, 0.280),
    "sad": (0.120, 0.410, 0.220),
    "weak": (0.210, 0.390, 0.260),
    "hate": (0.060, 0.850, 0.420),
    "angry": (0.140, 0.890, 0.510),
    "scam": (0.110, 0.780, 0.380),
    "galat": (0.160, 0.650, 0.330),
    "dhoka": (0.080, 0.820, 0.360),
    "shameful": (0.090, 0.760, 0.310),
    "bhikari": (0.100, 0.690, 0.240),
    "terrorist": (0.020, 0.960, 0.480),
    "scandal": (0.130, 0.810, 0.390),
    "ganda": (0.140, 0.550, 0.320),
    "kharab": (0.200, 0.600, 0.340),
    "bekar": (0.120, 0.480, 0.290),
    "chutiya": (0.050, 0.880, 0.350),
    "kamina": (0.070, 0.840, 0.370),
    "fraud": (0.080, 0.790, 0.370),
    "problem": (0.240, 0.620, 0.380),

    # High Arousal
    "urgent": (0.450, 0.880, 0.650),
    "action": (0.650, 0.780, 0.720),
    "jaldi": (0.500, 0.760, 0.580),
    "ladenge": (0.550, 0.890, 0.780),
    "protest": (0.350, 0.860, 0.620),

    # Low Arousal
    "chup": (0.480, 0.210, 0.450),
    "shant": (0.750, 0.180, 0.550),
    "slow": (0.420, 0.250, 0.380),
}


class NRCVADLexicon:
    """Symbolic NRC-VAD Affective Norms Lexicon.

    Maps tokens to continuous affective priors [Valence, Arousal] in [0, 1].
    Maintains an explicit boolean availability mask for missing/out-of-lexicon tokens.
    """

    def __init__(
        self,
        custom_dict: Optional[Dict[str, Union[Tuple[float, float], Tuple[float, float, float]]]] = None,
        custom_lexicon: Optional[Dict[str, Union[Tuple[float, float], Tuple[float, float, float]]]] = None,
    ):
        raw = custom_lexicon or custom_dict or EMBEDDED_NRC_VAD
        self.lexicon: Dict[str, Tuple[float, float]] = {}
        for k, v in raw.items():
            clean_k = self.normalize_token(k)
            if len(v) >= 2:
                self.lexicon[clean_k] = (float(v[0]), float(v[1]))

    def normalize_token(self, token: str) -> str:
        """Strip punctuation and convert to lowercase for lexicon lookup."""
        return re.sub(r"[^\w]", "", token).lower().strip()

    def lookup_token(self, token: str) -> Tuple[float, float, bool]:
        """Look up Valence and Arousal affective priors for a single token.

        Args:
            token: Raw or subword token string.

        Returns:
            Tuple of (valence_prior, arousal_prior, is_available).
            If token is missing, returns (0.0, 0.0, False).
        """
        clean = self.normalize_token(token)
        if clean in self.lexicon:
            v, a = self.lexicon[clean]
            return v, a, True
        return 0.0, 0.0, False

    def get_vad(self, token: str) -> Tuple[float, float, bool]:
        """Alias for lookup_token."""
        return self.lookup_token(token)

    def get_sentence_priors(self, tokens: List[str]) -> Tuple[List[List[float]], List[bool]]:
        """Compute affective prior vectors and availability mask for a token list.

        Args:
            tokens: List of token strings.

        Returns:
            Tuple of:
            - priors: List of [v_prior, a_prior] for each token
            - availability_mask: List of bool (True if in lexicon, False if missing)
        """
        priors: List[List[float]] = []
        mask: List[bool] = []

        for tok in tokens:
            v, a, is_avail = self.lookup_token(tok)
            priors.append([v, a])
            mask.append(is_avail)

        return priors, mask
