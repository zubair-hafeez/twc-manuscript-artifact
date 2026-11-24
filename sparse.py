"""Coefficient-level structural pruning (Sparse-ETWC v3) and its packed ablation.

The v2 implementation pruned whole candidate pairs but stored every retained
pair as a fixed (K-1, 2, 2) block, so pruned degrees were kept as zeros.
Here each retained (target i, source j, target side e, origin a) entry stores
only degrees lo..hi, where

  lo = delta_{j,a}(endpoint_i[e]; s_i)  (fewest original edges; lower degrees
       are structurally zero), and
  hi = K - 2 - beta_i(1-e)              (the pruning bound of THEORY.txt s.2).

Both bounds come from integer temporal reachability, never from numerical
zero tests. Construction processes sources in blocks of B columns per
history pass (B = q gives one pass, like the eager compiler; B = 1 gives the
per-source scan used by v2). The packed ablation uses the same engine and
layout but keeps every strictly time-ordered pair at full degree range, so
the two differ only in pruning.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def suffix_hops(src, dst, stamp, n, depth, candidates, nodes):
    """beta[i, side]: fewest original edges strictly after s_i from that endpoint to a row."""
    q = len(candidates)
    active = np.zeros((depth, n), np.bool_)
    for node in nodes:
        active[0, node] = True
    beta = np.full((q, 2), depth+1, np.int64)
    pos, end = q-1, len(src)
    while end > 0:
        start = end-1
        while start > 0 and stamp[start-1] == stamp[end-1]:
            start -= 1
        while pos >= 0 and candidates[pos] >= start:
            i = candidates[pos]
            for side in range(2):
                vertex = src[i] if side == 0 else dst[i]
                for k in range(depth):
                    if active[k, vertex]:
                        beta[pos, side] = k
                        break
            pos -= 1
        for k in range(depth-1, 0, -1):
            for x in range(start, end):
                u, v = src[x], dst[x]
                if active[k-1, v]:
                    active[k, u] = True
                if active[k-1, u]:
                    active[k, v] = True
        end = start
    return beta


@njit(cache=True)
def compact_block(src, dst, stamp, wev, depth, candidates, beta, j0, j1, prune, h, reach,
                  ent_i, ent_j, ent_e, ent_a, ent_lo, ent_hi, values):
    """One multi-column pass for sources j0..j1-1; h/reach columns 2*(j-j0)+a.

    Each source impulse is injected after its own timestamp group (strict
    time), exactly as in the eager compiler. Entries are emitted in target
    order and, within a target, in source order. h and reach must be zero on
    entry; the touched range is cleared on exit. Returns (entries, values, scanned).
    """
    q = len(candidates)
    if depth < 2 or j0 >= j1:
        return 0, 0, 0
    entries, nvalues = 0, 0
    first = np.searchsorted(stamp, stamp[candidates[j0]], side="left")
    start = first
    injected = j0  # sources j0..injected-1 carry impulses
    pos = j0
    while start < len(src) and pos < q:
        end = start+1
        while end < len(src) and stamp[end] == stamp[start]:
            end += 1
        group_pos = pos
        while pos < q and candidates[pos] < end:
            i = candidates[pos]
            for j in range(j0, injected):
                c = 2*(j-j0)
                for e in range(2):
                    vertex = src[i] if e == 0 else dst[i]
                    hi = depth-2-beta[pos, 1-e] if prune else depth-2
                    for a in range(2):
                        lo = 0
                        if prune:
                            lo = depth
                            for k in range(depth-1):
                                if reach[k, vertex, c+a]:
                                    lo = k
                                    break
                        if lo <= hi:
                            ent_i[entries] = pos
                            ent_j[entries] = j
                            ent_e[entries] = e
                            ent_a[entries] = a
                            ent_lo[entries] = lo
                            ent_hi[entries] = hi
                            for g in range(lo, hi+1):
                                values[nvalues] = h[g, vertex, c+a]
                                nvalues += 1
                            entries += 1
            pos += 1
        if pos >= q:
            break
        w = wev[start]
        width = 2*(injected-j0)
        for k in range(depth-2, 0, -1):
            for x in range(start, end):
                u, v = src[x], dst[x]
                for col in range(width):
                    h[k, u, col] += w*h[k-1, v, col]
                    h[k, v, col] += w*h[k-1, u, col]
                if prune:
                    for col in range(width):
                        if reach[k-1, v, col]:
                            reach[k, u, col] = True
                        if reach[k-1, u, col]:
                            reach[k, v, col] = True
        for j in range(max(group_pos, injected), min(pos, j1)):
            c = 2*(j-j0)
            h[0, src[candidates[j]], c] = 1.
            h[0, dst[candidates[j]], c+1] = 1.
            reach[0, src[candidates[j]], c] = True
            reach[0, dst[candidates[j]], c+1] = True
            injected = j+1
        start = end
    used = 2*(injected-j0)
    # Restore zeros: whichever is cheaper, the touched event range or the used slice.
    if 2*(start-first) < h.shape[1]:
        for x in range(first, start):
            for k in range(depth-1):
                for col in range(used):
                    h[k, src[x], col] = 0.
                    h[k, dst[x], col] = 0.
                    reach[k, src[x], col] = False
                    reach[k, dst[x], col] = False
    else:
        h[:, :, :used] = 0.
        reach[:, :, :used] = False
    return entries, nvalues, start-first


@njit(cache=True)
def compact_evaluate(prefix, suffix, weights, original, indptr, ent_j, ent_e, ent_a,
                     ent_lo, ent_hi, ent_off, values, selected, positions):
    """Exact causal recurrence over compact entries. positions is all -1 on entry and exit."""
    depth, d = prefix.shape[1], prefix.shape[-1]
    b = len(selected)
    for p in range(b):
        positions[selected[p]] = p
    injection = np.zeros((b, depth+1, 2, d))
    for pos in range(b):
        i = selected[pos]
        w = weights[i]
        for k in range(1, depth+1):
            for side in range(2):
                for col in range(d):
                    injection[pos, k, side, col] = -w*prefix[i, k-1, 1-side, col]
        for x in range(indptr[i], indptr[i+1]):
            earlier = positions[ent_j[x]]
            if earlier < 0 or earlier >= pos:
                continue
            side, origin = 1-ent_e[x], ent_a[x]
            lo, off = ent_lo[x], ent_off[x]
            for g in range(lo, ent_hi[x]+1):
                c = -w*values[off+g-lo]
                for level in range(1, depth-g):
                    k = g+1+level
                    for col in range(d):
                        injection[pos, k, side, col] += c*injection[earlier, level, origin, col]
    out = original.copy()
    for pos in range(b):
        i = selected[pos]
        for k in range(1, depth+1):
            for level in range(1, k+1):
                for row in range(out.shape[1]):
                    for side in range(2):
                        coefficient = suffix[i, k-level, row, side]
                        if coefficient != 0.:
                            for col in range(d):
                                out[k, row, col] += coefficient*injection[pos, level, side, col]
    for p in range(b):
        positions[selected[p]] = -1
    return out


@njit(cache=True)
def compact_batch(prefix, suffix, weights, original, indptr, ent_j, ent_e, ent_a, ent_lo,
                  ent_hi, ent_off, values, positions, hyps, lengths, out):
    for r in range(len(lengths)):
        out[r] = compact_evaluate(prefix, suffix, weights, original, indptr, ent_j, ent_e, ent_a,
                                  ent_lo, ent_hi, ent_off, values, hyps[r, :lengths[r]], positions)


@njit(cache=True)
def certified(indptr, ent_j, selected, mark):
    """True when no retained entry links two selected candidates. mark is all False on entry/exit."""
    for p in range(len(selected)):
        mark[selected[p]] = True
    ok = True
    for p in range(len(selected)):
        i = selected[p]
        for x in range(indptr[i], indptr[i+1]):
            if mark[ent_j[x]]:
                ok = False
                break
        if not ok:
            break
    for p in range(len(selected)):
        mark[selected[p]] = False
    return ok


@njit(cache=True)
def adaptive_batch(prefix, suffix, weights, original, indptr, ent_j, ent_e, ent_a, ent_lo,
                   ent_hi, ent_off, values, positions, mark, effects, hyps, lengths, out, used_ie):
    for r in range(len(lengths)):
        sel = hyps[r, :lengths[r]]
        if certified(indptr, ent_j, sel, mark):
            used_ie[r] = True
            out[r] = original
            for e in range(len(sel)):
                out[r] += effects[sel[e]]
        else:
            used_ie[r] = False
            out[r] = compact_evaluate(prefix, suffix, weights, original, indptr, ent_j, ent_e,
                                      ent_a, ent_lo, ent_hi, ent_off, values, sel, positions)
