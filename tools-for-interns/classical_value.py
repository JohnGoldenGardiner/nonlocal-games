"""Classical value of a two-player nonlocal game by enumerating deterministic strategies."""

import numpy as np


def _all_sums(w):
    """w[a, x, b, y] -> t[a, x, s] = sum_y w[a, x, s_y, y] over every assignment s of an answer b to each y."""
    t = w[..., 0]
    for y in range(1, w.shape[-1]):
        t = (t[..., :, None] + w[:, :, None, :, y]).reshape(*t.shape[:2], -1)
    return t


def classical_value(pi, V, block=2**18):
    """Max over deterministic strategies of sum_{x,y} pi[x, y] V[x, y, a(x), b(y)].

    pi has shape (X, Y); V has shape (X, Y, A, B) and may hold any real payoffs.
    One player's deterministic strategies are enumerated and the other best-responds question by question.
    """
    w = np.asarray(pi[:, :, None, None] * V, dtype=float)  # w[x, y, a, b]
    X, Y, A, B = w.shape
    # Enumerating Bob costs ~ A*X*B**Y, enumerating Alice ~ B*Y*A**X: enumerate the cheaper side.
    if B * Y * A**X < A * X * B**Y:
        w, A, B, X, Y = w.transpose(1, 0, 3, 2), B, A, Y, X
    w = w.transpose(2, 0, 3, 1)  # w[a, x, b, y]: Alice's axes first so reductions run over contiguous slabs

    # Split Bob's questions: the last k are enumerated in one vectorized block, the rest in an outer loop.
    k = 1
    while k < Y and A * X * B ** (k + 1) <= block:
        k += 1
    inner = _all_sums(w[..., Y - k :])  # (A, X, B**k)
    outer = _all_sums(w[..., : Y - k]) if k < Y else np.zeros((A, X, 1))  # (A, X, B**(Y-k))

    buf = np.empty_like(inner)
    best = -np.inf
    for p in range(outer.shape[-1]):
        np.add(inner, outer[:, :, p, None], out=buf)
        best = max(best, buf.max(axis=0).sum(axis=0).max())  # Alice best-responds to each x, then sum over x
    return float(best)
