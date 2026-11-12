"""No-signalling value of a two-player nonlocal game (an upper bound on the quantum value)."""

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


def _diff(n):
    """(n-1, n) matrix whose rows are e_i - e_{i+1}."""
    return sp.eye(n - 1, n) - sp.eye(n - 1, n, k=1)


def nonsignaling_value(pi, V):
    """Max of sum pi[x, y] V[x, y, a, b] p[x, y, a, b] over no-signalling behaviours p, as a sparse LP solved by HiGHS.

    pi has shape (X, Y); V has shape (X, Y, A, B) and may hold any real payoffs.
    The variable is p[x, y, a, b] >= 0 flattened in C order, so constraints are Kronecker products over (x, y, a, b).
    """
    w = np.asarray(pi[:, :, None, None] * V, dtype=float)
    X, Y, A, B = w.shape
    one = lambda n: np.ones((1, n))
    kron = lambda *ms: sp.kron(sp.kron(ms[0], ms[1]), sp.kron(ms[2], ms[3]))
    # A linearly independent description of the no-signalling polytope:
    #   sum_{a,b} p[x, y, a, b] = 1                       for every (x, y),
    #   sum_b p[x, y, a, b] = sum_b p[x, y+1, a, b]       for a < A-1 (the last a then follows from normalization),
    #   sum_a p[x, y, a, b] = sum_a p[x+1, y, a, b]       for b < B-1.
    a_eq = sp.vstack([
        kron(sp.eye(X), sp.eye(Y), one(A), one(B)),
        kron(sp.eye(X), _diff(Y), sp.eye(A - 1, A), one(B)),
        kron(_diff(X), sp.eye(Y), one(A), sp.eye(B - 1, B)),
    ]).tocsr()
    b_eq = np.zeros(a_eq.shape[0])
    b_eq[: X * Y] = 1
    res = linprog(-w.ravel(), A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    return -res.fun
