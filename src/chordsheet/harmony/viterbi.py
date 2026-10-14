"""HMM smoothing of per-beat chord scores."""

from __future__ import annotations

import numpy as np


def viterbi(scores: np.ndarray, self_transition: float = 0.85, emission_scale: float = 20.0) -> np.ndarray:
    """Best state path through `scores` (n_frames, n_states).

    Emissions are `emission_scale * scores` used as log-probabilities; the
    transition matrix keeps a state with `self_transition` and spreads the rest
    uniformly. Higher self_transition => fewer, longer chords."""
    n, k = scores.shape
    if n == 0:
        return np.zeros(0, dtype=int)
    log_stay = np.log(self_transition)
    log_move = np.log((1 - self_transition) / (k - 1))
    emit = emission_scale * scores

    delta = emit[0].copy()
    back = np.zeros((n, k), dtype=int)
    for t in range(1, n):
        # transition matrix is uniform off-diagonal, so the best "move" source is the global argmax
        best_prev = int(delta.argmax())
        move = delta[best_prev] + log_move
        stay = delta + log_stay
        use_stay = stay >= move
        back[t] = np.where(use_stay, np.arange(k), best_prev)
        delta = np.where(use_stay, stay, move) + emit[t]

    path = np.zeros(n, dtype=int)
    path[-1] = int(delta.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
