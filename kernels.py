"""Equal-engine Numba kernels for eager and lazy exact deletion evaluation.

Every evaluator here prepares candidate transfers in the dense layout used by
the v1 compiler, transfers[i, degree, target_side, 2*j+origin], and scores a
hypothesis with the unchanged v1 kernel ``evaluate_circuit``. The evaluators
therefore differ only in WHEN and HOW transfers are produced:

* eager (ETWC): one multi-column pass over the candidate time span;
* source-batched lazy (SB-Lazy): one two-column pass per requested source;
* pair lazy (Pair-Lazy): one two-column interval pass per requested pair.

All passes share precomputed per-event decay weights and the same propagation
loop. Work buffers are allocated once and restored to zero after each pass,
by re-scanning the touched event range or by zeroing the buffer, whichever
touches fewer entries.
"""
import numpy as np
from numba import njit
from revision_circuit import evaluate_circuit


@njit(cache=True)
def event_weights(stamp, query_time, rate):
    # Identical expression to the v1 per-group weight, so values are bitwise equal.
    out = np.empty(len(stamp))
    for e in range(len(stamp)):
        out[e] = np.exp(-rate*(query_time-stamp[e]))
    return out


@njit(cache=True)
def eager_transfers(src, dst, stamp, wev, n, depth, candidates):
    """Multi-column eager pass (same recurrence as v1 candidate_transfers)."""
    q = len(candidates)
    transfers = np.zeros((q, max(depth-1, 0), 2, 2*q))
    if depth < 2 or q == 0:
        return transfers
    h = np.zeros((depth-1, n, 2*q))
    pos, active = 0, 0
    start = np.searchsorted(stamp, stamp[candidates[0]], side="left")
    while start < len(src) and start <= candidates[-1]:
        end = start+1
        while end < len(src) and stamp[end] == stamp[start]:
            end += 1
        next_pos = pos
        while next_pos < q and candidates[next_pos] < end:
            i = candidates[next_pos]
            for k in range(depth-1):
                for col in range(active):
                    transfers[next_pos, k, 0, col] = h[k, src[i], col]
                    transfers[next_pos, k, 1, col] = h[k, dst[i], col]
            next_pos += 1
        w = wev[start]
        for k in range(depth-2, 0, -1):
            for x in range(start, end):
                u, v = src[x], dst[x]
                for col in range(active):
                    h[k, u, col] += w*h[k-1, v, col]
                    h[k, v, col] += w*h[k-1, u, col]
        for j in range(pos, next_pos):
            i = candidates[j]
            h[0, src[i], 2*j] = 1.
            h[0, dst[i], 2*j+1] = 1.
        pos, active, start = next_pos, 2*next_pos, end
    return transfers


@njit(cache=True)
def _clear(src, dst, h, first, last, a, b):
    if 2*(last-first) >= h.shape[1]:
        h[:] = 0.
        return
    for x in range(first, last):
        for k in range(h.shape[0]):
            for col in range(2):
                h[k, src[x], col] = 0.
                h[k, dst[x], col] = 0.
    for k in range(h.shape[0]):
        for col in range(2):
            h[k, a, col] = 0.
            h[k, b, col] = 0.


@njit(cache=True)
def source_scan(src, dst, stamp, wev, depth, candidates, j, transfers, h):
    """Fill transfers from source j to every later candidate; h is zero on entry and exit."""
    q = len(candidates)
    e = candidates[j]
    if depth < 2:
        return 0
    h[0, src[e], 0] = 1.
    h[0, dst[e], 1] = 1.
    first = np.searchsorted(stamp, stamp[e], side="right")
    start = first
    pos = j+1
    while pos < q and stamp[candidates[pos]] == stamp[e]:
        pos += 1
    while start < len(src) and pos < q:
        end = start+1
        while end < len(src) and stamp[end] == stamp[start]:
            end += 1
        while pos < q and candidates[pos] < end:
            i = candidates[pos]
            for k in range(depth-1):
                for a in range(2):
                    transfers[pos, k, 0, 2*j+a] = h[k, src[i], a]
                    transfers[pos, k, 1, 2*j+a] = h[k, dst[i], a]
            pos += 1
        if pos >= q:
            break
        w = wev[start]
        for k in range(depth-2, 0, -1):
            for x in range(start, end):
                u, v = src[x], dst[x]
                for a in range(2):
                    h[k, u, a] += w*h[k-1, v, a]
                    h[k, v, a] += w*h[k-1, u, a]
        start = end
    scanned = start-first
    _clear(src, dst, h, first, start, src[e], dst[e])
    return scanned


@njit(cache=True)
def pair_scan(src, dst, stamp, wev, depth, candidates, j, pos_i, transfers, h):
    """Fill the single (j -> pos_i) transfer block from the strict interval."""
    e, target = candidates[j], candidates[pos_i]
    if depth < 2 or stamp[e] >= stamp[target]:
        return 0
    h[0, src[e], 0] = 1.
    h[0, dst[e], 1] = 1.
    first = np.searchsorted(stamp, stamp[e], side="right")
    stop = np.searchsorted(stamp, stamp[target], side="left")
    start = first
    while start < stop:
        end = start+1
        while end < stop and stamp[end] == stamp[start]:
            end += 1
        w = wev[start]
        for k in range(depth-2, 0, -1):
            for x in range(start, end):
                u, v = src[x], dst[x]
                for a in range(2):
                    h[k, u, a] += w*h[k-1, v, a]
                    h[k, v, a] += w*h[k-1, u, a]
        start = end
    for k in range(depth-1):
        for a in range(2):
            transfers[pos_i, k, 0, 2*j+a] = h[k, src[target], a]
            transfers[pos_i, k, 1, 2*j+a] = h[k, dst[target], a]
    _clear(src, dst, h, first, stop, src[e], dst[e])
    return stop-first


@njit(cache=True)
def eager_batch(prefix, transfers, suffix, weights, original, hyps, lengths, out):
    for r in range(len(lengths)):
        out[r] = evaluate_circuit(prefix, transfers, suffix, weights, hyps[r, :lengths[r]], original)


@njit(cache=True)
def source_lazy_batch(src, dst, stamp, wev, depth, candidates, prefix, transfers, suffix,
                      weights, original, filled, h, hyps, lengths, out):
    misses, scanned = 0, 0
    for r in range(len(lengths)):
        b = lengths[r]
        for e in range(b-1):
            j = hyps[r, e]
            if not filled[j]:
                scanned += source_scan(src, dst, stamp, wev, depth, candidates, j, transfers, h)
                filled[j] = True
                misses += 1
        out[r] = evaluate_circuit(prefix, transfers, suffix, weights, hyps[r, :b], original)
    return misses, scanned


@njit(cache=True)
def pair_lazy_batch(src, dst, stamp, wev, depth, candidates, prefix, transfers, suffix,
                    weights, original, filled, h, hyps, lengths, out):
    misses, scanned = 0, 0
    for r in range(len(lengths)):
        b = lengths[r]
        for later in range(1, b):
            i = hyps[r, later]
            for earlier in range(later):
                j = hyps[r, earlier]
                if not filled[i, j]:
                    scanned += pair_scan(src, dst, stamp, wev, depth, candidates, j, i, transfers, h)
                    filled[i, j] = True
                    misses += 1
        out[r] = evaluate_circuit(prefix, transfers, suffix, weights, hyps[r, :b], original)
    return misses, scanned


@njit(cache=True)
def independent_batch(effects, original, hyps, lengths, out):
    for r in range(len(lengths)):
        out[r] = original
        for e in range(lengths[r]):
            out[r] += effects[hyps[r, e]]
