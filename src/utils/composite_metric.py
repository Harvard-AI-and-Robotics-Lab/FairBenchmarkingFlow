"""
Composite scoring metric for generative model evaluation.

MinMax Harmonic Mean combining FID (lower is better),
IS, CLIP Score, and Pick Score (all higher is better).
"""

import numpy as np


def _minmax_normalize(
    values: np.ndarray,
    lower_is_better: bool,
    lo: float | None = None,
    hi: float | None = None,
) -> np.ndarray:
    """Normalize to [0, 1] so that the best model scores 1.

    If ``lo``/``hi`` are provided they are used instead of the data min/max,
    which allows applying training-set bounds to a new evaluation set.
    """
    lo = float(values.min()) if lo is None else float(lo)
    hi = float(values.max()) if hi is None else float(hi)
    if hi == lo:
        return np.ones_like(values)
    if lower_is_better:
        return (hi - values) / (hi - lo)   # FID: max→0, min→1
    else:
        return (values - lo) / (hi - lo)   # IS/CLIP/Pick: min→0, max→1


# Keys used in the bounds dict (must stay in sync with the script).
_BOUND_KEYS = ("FID", "IS", "CLIP Score", "Pick Score")


def minmax_harmonic_mean(
    fid_scores: list[float],
    is_scores: list[float],
    clip_scores: list[float],
    pick_scores: list[float],
    eps: float = 1e-3,
    bounds: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Min-max normalize within the set of models, then combine with harmonic mean.

    u_FID(m)  = (max(FID) - FID(m))   / (max(FID)  - min(FID))
    u_IS(m)   = (IS(m)   - min(IS))   / (max(IS)   - min(IS))
    u_CLIP(m) = (CLIP(m) - min(CLIP)) / (max(CLIP) - min(CLIP))
    u_PICK(m) = (PICK(m) - min(PICK)) / (max(PICK) - min(PICK))

    S(m) = 4 / (1/(eps+u_FID) + 1/(eps+u_IS) + 1/(eps+u_CLIP) + 1/(eps+u_PICK))

    Parameters
    ----------
    bounds:
        Optional dict mapping each metric name to ``{"min": float, "max": float}``.
        When supplied, the stored bounds are used for normalization instead of
        recomputing them from the current data (useful for generalizing scores
        to new models without shifting the scale).

    Returns
    -------
    scores : np.ndarray
    bounds_used : dict
        The ``{"min", "max"}`` bounds actually used for each metric, suitable
        for serialization and later reuse via the ``bounds`` parameter.
    """
    fid  = np.array(fid_scores,  dtype=float)
    is_  = np.array(is_scores,   dtype=float)
    clip = np.array(clip_scores, dtype=float)
    pick = np.array(pick_scores, dtype=float)

    def _get(key, arr):
        b = (bounds or {}).get(key, {})
        return b.get("min"), b.get("max"), arr.min(), arr.max()

    def _norm(key, arr, lower_is_better):
        pre_lo, pre_hi, data_lo, data_hi = _get(key, arr)
        lo = pre_lo if pre_lo is not None else data_lo
        hi = pre_hi if pre_hi is not None else data_hi
        return _minmax_normalize(arr, lower_is_better, lo=lo, hi=hi), float(lo), float(hi)

    u_fid,  fid_lo,  fid_hi  = _norm("FID",        fid,  lower_is_better=True)
    u_is,   is_lo,   is_hi   = _norm("IS",          is_,  lower_is_better=False)
    u_clip, clip_lo, clip_hi = _norm("CLIP Score",  clip, lower_is_better=False)
    u_pick, pick_lo, pick_hi = _norm("Pick Score",  pick, lower_is_better=False)

    scores = 4.0 / (
        1.0 / (eps + u_fid)
        + 1.0 / (eps + u_is)
        + 1.0 / (eps + u_clip)
        + 1.0 / (eps + u_pick)
    )

    bounds_used = {
        "FID":        {"min": fid_lo,  "max": fid_hi},
        "IS":         {"min": is_lo,   "max": is_hi},
        "CLIP Score": {"min": clip_lo, "max": clip_hi},
        "Pick Score": {"min": pick_lo, "max": pick_hi},
    }

    return scores, bounds_used
