"""Optimized first-order control; deliberately omits edit interactions."""
from dataclasses import dataclass
import numpy as np
from numba import njit
from revision_circuit import forward_index, backward_query_index


@njit(cache=True)
def single_effects(prefix, suffix, weights):
    q, depth, _, d = prefix.shape
    out = np.zeros((q, depth+1, suffix.shape[2], d))
    for i in range(q):
        for k in range(1, depth+1):
            for level in range(1, k+1):
                for row in range(suffix.shape[2]):
                    for side in range(2):
                        coefficient = -weights[i]*suffix[i, k-level, row, side]
                        if coefficient != 0.:
                            for col in range(d):
                                out[i, k, row, col] += coefficient*prefix[i, level-1, 1-side, col]
    return out


@dataclass
class IndependentDeletionEffects:
    src: np.ndarray
    dst: np.ndarray
    stamp: np.ndarray
    base: np.ndarray
    depth: int
    query_time: float
    rate: float
    candidates: np.ndarray
    nodes: np.ndarray

    def __post_init__(self):
        self.candidates = np.sort(np.asarray(self.candidates, np.int64))
        self.nodes = np.asarray(self.nodes, np.int64)
        h, prefix = forward_index(self.src, self.dst, self.stamp, self.base, self.depth,
                                   self.query_time, self.rate, self.candidates)
        self.original = h[:, self.nodes].copy()
        suffix = backward_query_index(self.src, self.dst, self.stamp, len(self.base), self.depth,
                                        self.query_time, self.rate, self.candidates, self.nodes)
        weights = np.exp(-self.rate*(self.query_time-self.stamp[self.candidates]))
        self.effects = single_effects(prefix, suffix, weights)

    def evaluate(self, event_ids):
        ids = np.sort(np.asarray(event_ids, np.int64))
        selected = np.searchsorted(self.candidates, ids)
        if (np.any(selected >= len(self.candidates))
                or not np.array_equal(self.candidates[selected], ids)
                or len(np.unique(ids)) != len(ids)):
            raise ValueError("Invalid deletion")
        return self.original+self.effects[selected].sum(axis=0)
