"""Uniform Python wrappers around the v3 kernels.

All evaluators share the same validation, prefix/suffix construction, decay
weights, hypothesis encoding and numeric scoring kernel, so measured
differences come from transfer preparation and storage policy only.
Hypotheses are passed as sorted candidate POSITIONS (0..q-1), encoded once
outside every timed region by ``encode``.
"""
import numpy as np
from oracles import validate
from revision_circuit import (
    forward_index, backward_query_index)
from independent_effects import single_effects
import kernels as K
import sparse as S


def encode(candidates, hypotheses):
    """Event-id tuples -> padded sorted candidate positions and lengths."""
    candidates = np.sort(np.asarray(candidates, np.int64))
    width = max([len(h) for h in hypotheses] + [1])
    hyps = np.zeros((len(hypotheses), width), np.int64)
    lengths = np.zeros(len(hypotheses), np.int64)
    for r, ids in enumerate(hypotheses):
        ids = np.sort(np.asarray(ids, np.int64))
        pos = np.searchsorted(candidates, ids)
        if (len(np.unique(ids)) != len(ids) or np.any(pos >= len(candidates))
                or not np.array_equal(candidates[pos], ids)):
            raise ValueError("Deletion outside candidate set or duplicate deletion")
        hyps[r, :len(ids)] = pos
        lengths[r] = len(ids)
    return hyps, lengths


class _Base:
    def __init__(self, src, dst, stamp, base, depth, query_time, rate, candidates, nodes):
        src, dst, stamp, base, candidates, nodes = validate(
            src, dst, stamp, base, depth, query_time, rate, candidates, nodes)
        self.src, self.dst, self.stamp = src, dst, stamp
        self.n, self.depth, self.query_time, self.rate = len(base), depth, query_time, rate
        self.candidates, self.nodes, self.d = candidates, nodes, base.shape[1]
        h, self.prefix = forward_index(src, dst, stamp, base, depth, query_time, rate, candidates)
        self.original = h[:, nodes].copy()
        del h
        self.suffix = backward_query_index(src, dst, stamp, len(base), depth, query_time, rate,
                                           candidates, nodes)
        self.weights = np.exp(-rate*(query_time-stamp[candidates]))

    def out_buffer(self, count):
        return np.empty((count,)+self.original.shape)

    def evaluate(self, event_ids):
        hyps, lengths = encode(self.candidates, [tuple(event_ids)])
        out = self.out_buffer(1)
        self.batch(hyps, lengths, out)
        return out[0]

    def shared_bytes(self):
        return sum(a.nbytes for a in (self.prefix, self.suffix, self.weights, self.original))


class Eager(_Base):
    """ETWC: eager multi-column transfer compilation."""
    name = "eager_etwc"

    def __init__(self, *args):
        super().__init__(*args)
        wev = K.event_weights(self.stamp, self.query_time, self.rate)
        self.transfers = K.eager_transfers(self.src, self.dst, self.stamp, wev, self.n,
                                           self.depth, self.candidates)

    def batch(self, hyps, lengths, out):
        K.eager_batch(self.prefix, self.transfers, self.suffix, self.weights, self.original,
                      hyps, lengths, out)

    def storage(self):
        q, k = len(self.candidates), max(self.depth-1, 0)
        stamps = self.stamp[self.candidates]
        ordered = int(np.sum(stamps[None, :] < stamps[:, None]))  # (i later than j)
        return {"shared_bytes": self.shared_bytes(), "transfer_bytes": self.transfers.nbytes,
                "transfer_coefficients_structurally_possible": ordered*k*4,
                "transfer_coefficients_allocated": q*k*2*2*q, "index_bytes": 0}


class _Lazy(_Base):
    def __init__(self, *args):
        super().__init__(*args)
        q = len(self.candidates)
        self.wev = K.event_weights(self.stamp, self.query_time, self.rate)
        self.transfers = np.zeros((q, max(self.depth-1, 0), 2, 2*q))
        self.h = np.zeros((max(self.depth-1, 1), self.n, 2))
        self.misses, self.scanned = 0, 0


class SourceLazy(_Lazy):
    """SB-Lazy: one two-column history pass per first-requested source."""
    name = "source_lazy"

    def __init__(self, *args):
        super().__init__(*args)
        self.filled = np.zeros(len(self.candidates), np.bool_)

    def batch(self, hyps, lengths, out):
        m, s = K.source_lazy_batch(self.src, self.dst, self.stamp, self.wev, self.depth,
                                   self.candidates, self.prefix, self.transfers, self.suffix,
                                   self.weights, self.original, self.filled, self.h, hyps,
                                   lengths, out)
        self.misses += m
        self.scanned += s

    def storage(self):
        k = max(self.depth-1, 0)
        q = len(self.candidates)
        computed = int(self.filled.sum())
        return {"shared_bytes": self.shared_bytes(), "transfer_bytes_computed": computed*q*k*4*8,
                "transfer_bytes_allocated": self.transfers.nbytes,
                "resident_event_bytes": self.src.nbytes+self.dst.nbytes+self.stamp.nbytes+self.wev.nbytes,
                "sources_computed": computed}


class PairLazy(_Lazy):
    """Pair-Lazy: one two-column interval pass per first-requested ordered pair."""
    name = "pair_lazy"

    def __init__(self, *args):
        super().__init__(*args)
        q = len(self.candidates)
        self.filled = np.zeros((q, q), np.bool_)

    def batch(self, hyps, lengths, out):
        m, s = K.pair_lazy_batch(self.src, self.dst, self.stamp, self.wev, self.depth,
                                 self.candidates, self.prefix, self.transfers, self.suffix,
                                 self.weights, self.original, self.filled, self.h, hyps,
                                 lengths, out)
        self.misses += m
        self.scanned += s

    def storage(self):
        k = max(self.depth-1, 0)
        computed = int(self.filled.sum())
        return {"shared_bytes": self.shared_bytes(), "transfer_bytes_computed": computed*k*4*8,
                "transfer_bytes_allocated": self.transfers.nbytes,
                "resident_event_bytes": self.src.nbytes+self.dst.nbytes+self.stamp.nbytes+self.wev.nbytes,
                "pairs_computed": computed}


class Compact(_Base):
    """Sparse-ETWC v3 (prune=True) or its packed, unpruned ablation (prune=False)."""

    def __init__(self, *args, prune=True, block=None):
        super().__init__(*args)
        self.prune = prune
        self.name = "sparse_compact" if prune else "packed_compact"
        q, depth = len(self.candidates), self.depth
        self.block = max(1, q if block is None else min(block, q)) if q else 1
        wev = K.event_weights(self.stamp, self.query_time, self.rate)
        self.beta = S.suffix_hops(self.src, self.dst, self.stamp, self.n, depth,
                                  self.candidates, self.nodes)
        width = 2*self.block
        h = np.zeros((max(depth-1, 1), self.n, width))
        reach = np.zeros((max(depth-1, 1), self.n, width), np.bool_)
        cap = 4*q*self.block
        buf_i, buf_j, buf_e, buf_a = (np.empty(cap, np.int64) for _ in range(4))
        buf_lo, buf_hi = np.empty(cap, np.int64), np.empty(cap, np.int64)
        buf_v = np.empty(cap*max(depth-1, 1))
        rows = []
        self.scanned = 0
        for j0 in range(0, q, self.block):
            ne, nv, sc = S.compact_block(self.src, self.dst, self.stamp, wev, depth, self.candidates,
                                         self.beta, j0, min(j0+self.block, q), prune, h, reach,
                                         buf_i, buf_j, buf_e, buf_a, buf_lo, buf_hi, buf_v)
            self.scanned += sc
            if ne:
                rows.append((buf_j[:ne].copy(), buf_i[:ne].copy(), buf_e[:ne].copy(), buf_a[:ne].copy(),
                             buf_lo[:ne].copy(), buf_hi[:ne].copy(), buf_v[:nv].copy()))
        self.workspace_bytes = h.nbytes+reach.nbytes+sum(
            b.nbytes for b in (buf_i, buf_j, buf_e, buf_a, buf_lo, buf_hi, buf_v))
        if rows:
            ent_j = np.concatenate([r[0] for r in rows]); ent_i = np.concatenate([r[1] for r in rows])
            ent_e = np.concatenate([r[2] for r in rows]); ent_a = np.concatenate([r[3] for r in rows])
            ent_lo = np.concatenate([r[4] for r in rows]); ent_hi = np.concatenate([r[5] for r in rows])
            vals = [r[6] for r in rows]
            lengths = ent_hi-ent_lo+1
            source_off = np.cumsum([0]+[len(v) for v in vals])[:-1]
            within = np.concatenate([np.r_[0, np.cumsum(r[5]-r[4]+1)[:-1]] for r in rows])
            flat = np.concatenate(vals)
            raw_off = np.repeat(source_off, [len(r[0]) for r in rows])+within
            order = np.lexsort((ent_a, ent_e, ent_j, ent_i))
            ent_j, ent_i, ent_e, ent_a = ent_j[order], ent_i[order], ent_e[order], ent_a[order]
            ent_lo, ent_hi, lengths, raw_off = ent_lo[order], ent_hi[order], lengths[order], raw_off[order]
            self.ent_off = np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64)
            gather = np.repeat(raw_off-self.ent_off, lengths)+np.arange(int(lengths.sum()))
            self.values = flat[gather]
        else:
            ent_j = ent_i = ent_e = ent_a = ent_lo = ent_hi = np.empty(0, np.int64)
            self.values, self.ent_off = np.empty(0), np.empty(0, np.int64)
        self.indptr = np.searchsorted(ent_i, np.arange(q+1)).astype(np.int64)
        self.ent_j = ent_j.astype(np.int32)
        self.ent_e, self.ent_a = ent_e.astype(np.int8), ent_a.astype(np.int8)
        self.ent_lo, self.ent_hi = ent_lo.astype(np.int8), ent_hi.astype(np.int8)
        self.positions = np.full(q, -1, np.int64)
        self.mark = np.zeros(q, np.bool_)
        self.pairs = len(set(zip(ent_i.tolist(), ent_j.tolist())))

    def batch(self, hyps, lengths, out):
        S.compact_batch(self.prefix, self.suffix, self.weights, self.original, self.indptr,
                        self.ent_j, self.ent_e, self.ent_a, self.ent_lo, self.ent_hi, self.ent_off,
                        self.values, self.positions, hyps, lengths, out)

    def independent_exact(self, event_ids):
        if not self.prune:
            raise NotImplementedError("The packed ablation keeps every ordered pair and has no certificate")
        hyps, lengths = encode(self.candidates, [tuple(event_ids)])
        return bool(S.certified(self.indptr, self.ent_j, hyps[0, :lengths[0]], self.mark))

    def certificate_batch(self, hyps, lengths):
        return np.array([S.certified(self.indptr, self.ent_j, hyps[r, :lengths[r]], self.mark)
                         for r in range(len(lengths))], bool)

    def storage(self):
        index = sum(a.nbytes for a in (self.indptr, self.ent_j, self.ent_e, self.ent_a,
                                       self.ent_lo, self.ent_hi, self.ent_off))
        return {"shared_bytes": self.shared_bytes(), "coefficient_bytes": self.values.nbytes,
                "coefficients": len(self.values), "entries": len(self.ent_j),
                "index_bytes": index, "beta_bytes": self.beta.nbytes, "padding_bytes": 0,
                "retained_pairs": self.pairs, "construction_workspace_bytes": self.workspace_bytes}


class Independent(_Base):
    """Independent Effects (IE): sum of exact singleton effects; exact only when certified."""
    name = "independent_effects"

    def __init__(self, *args):
        super().__init__(*args)
        self.effects = single_effects(self.prefix, self.suffix, self.weights)

    def batch(self, hyps, lengths, out):
        K.independent_batch(self.effects, self.original, hyps, lengths, out)

    def storage(self):
        return {"shared_bytes": self.shared_bytes(), "effect_bytes": self.effects.nbytes}


class Adaptive(Compact):
    """Certificate-aware exact evaluator: IE when certified, Sparse-ETWC otherwise."""
    name = "adaptive"

    def __init__(self, *args):
        super().__init__(*args, prune=True)
        self.name = "adaptive"
        self.effects = single_effects(self.prefix, self.suffix, self.weights)
        self.used_ie = 0

    def batch(self, hyps, lengths, out):
        used = np.zeros(len(lengths), np.bool_)
        S.adaptive_batch(self.prefix, self.suffix, self.weights, self.original, self.indptr,
                         self.ent_j, self.ent_e, self.ent_a, self.ent_lo, self.ent_hi,
                         self.ent_off, self.values, self.positions, self.mark, self.effects,
                         hyps, lengths, out, used)
        self.used_ie += int(used.sum())

    def storage(self):
        out = super().storage()
        out["effect_bytes"] = self.effects.nbytes
        return out


METHODS = {"eager_etwc": Eager, "source_lazy": SourceLazy, "pair_lazy": PairLazy,
           "sparse_compact": lambda *a: Compact(*a, prune=True),
           "packed_compact": lambda *a: Compact(*a, prune=False),
           "sparse_per_source": lambda *a: Compact(*a, prune=True, block=1),
           "independent_effects": Independent, "adaptive": Adaptive}
