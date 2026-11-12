"""See-saw lower bound on the quantum value of a two-player nonlocal game [Liang & Doherty, PRA 75, 042103 (2007)].

Alice's side is an assemblage rho[x, a] = Tr_A[(A_a^x (x) 1) psi] on Bob's d-dimensional space, with
sum_a rho[x, a] = tau for every x and Tr tau = 1; any such assemblage is realisable with a d-dimensional Alice.
The value is sum_{x,y,a,b} w[x,y,a,b] Tr(rho[x,a] B[y,b]) with w = pi * V. We alternate:
  * Alice step (SDP): maximise sum_{x,a} Tr(C[x,a] rho[x,a]), C[x,a] = sum_{y,b} w[x,y,a,b] B[y,b].
  * Bob step: maximise sum_{y,b} Tr(D[y,b] B[y,b]) over POVMs, D[y,b] = sum_{x,a} w[x,y,a,b] rho[x,a].
    With two outcomes this is solved exactly by projecting onto the positive part of D[y,0] - D[y,1].
Both SDPs are built once with the coefficients as cvxpy Parameters, so each step only re-solves.
"""

import cvxpy as cp
import numpy as np


def _make_psd(ops):
    """Shift ops[i, k] by delta * 1 so all are PSD; with the last outcome eliminated, every sum over k is exact."""
    delta = max(0.0, -np.linalg.eigvalsh(ops).min())
    return ops + delta * np.eye(ops.shape[-1]), ops.shape[1] * delta


def _alice_sdp(X, A, d):
    """SDP over assemblages; the last answer is eliminated as rho[x, A-1] = tau - sum_{a<A-1} rho[x, a]."""
    tau = cp.Variable((d, d), hermitian=True)
    r = [[cp.Variable((d, d), hermitian=True) for _ in range(A - 1)] for _ in range(X)]
    g = [[cp.Parameter((d, d), complex=True) for _ in range(A - 1)] for _ in range(X)]
    g_tau = cp.Parameter((d, d), complex=True)
    rho = [r[x] + [tau - sum(r[x])] for x in range(X)]
    obj = cp.trace(g_tau @ tau) + sum(cp.trace(g[x][a] @ r[x][a]) for x in range(X) for a in range(A - 1))
    cons = [cp.trace(tau) == 1] + [op >> 0 for row in rho for op in row]
    prob = cp.Problem(cp.Maximize(cp.real(obj)), cons)

    def solve(c, **kw):  # c[x, a] -> rho[x, a]
        g_tau.value = c[:, -1].sum(axis=0)
        for x in range(X):
            for a in range(A - 1):
                g[x][a].value = c[x, a] - c[x, -1]
        prob.solve(**kw)
        # Repair solver round-off so rho is an exactly valid assemblage: PSD, sum_a rho[x, a] = tau, Tr tau = 1.
        rho_val, shift = _make_psd(np.array([[op.value for op in row] for row in rho]))
        return rho_val / (np.trace(tau.value).real + d * shift)

    return solve


def _bob_sdp(Y, B, d):
    """SDP over POVMs; the last outcome is eliminated as B[y, B-1] = 1 - sum_{b<B-1} B[y, b]."""
    v = [[cp.Variable((d, d), hermitian=True) for _ in range(B - 1)] for _ in range(Y)]
    g = [[cp.Parameter((d, d), complex=True) for _ in range(B - 1)] for _ in range(Y)]
    povm = [v[y] + [np.eye(d) - sum(v[y])] for y in range(Y)]
    obj = sum(cp.trace(g[y][b] @ v[y][b]) for y in range(Y) for b in range(B - 1))
    prob = cp.Problem(cp.Maximize(cp.real(obj)), [op >> 0 for row in povm for op in row])

    def solve(dmat, **kw):  # dmat[y, b] -> (value, B[y, b])
        for y in range(Y):
            for b in range(B - 1):
                g[y][b].value = dmat[y, b] - dmat[y, -1]
        prob.solve(**kw)
        # Repair solver round-off so the POVMs are exactly valid, then score them exactly.
        ops, shift = _make_psd(np.array([[np.asarray(op.value) for op in row] for row in povm]))
        ops /= 1 + shift
        return np.einsum("ybij,ybji->", dmat, ops).real, ops

    return solve


def _bob_two_outcome(dmat, **_):
    """Exact Bob step for two outcomes: B[y, 0] = projector onto the positive eigenspace of D[y, 0] - D[y, 1]."""
    diff = dmat[:, 0] - dmat[:, 1]
    evals, evecs = np.linalg.eigh((diff + diff.conj().transpose(0, 2, 1)) / 2)
    pos = evecs * (evals > 0)[:, None, :]
    p0 = pos @ pos.conj().transpose(0, 2, 1)
    val = np.trace(dmat[:, 1], axis1=1, axis2=2).real.sum() + np.clip(evals, 0, None).sum()
    return val, np.stack([p0, np.eye(diff.shape[-1]) - p0], axis=1)


def _random_projective(Y, B, d, rng):
    """Haar-random orthonormal basis per question, basis vector j assigned to outcome j % B."""
    z = rng.normal(size=(Y, d, d)) + 1j * rng.normal(size=(Y, d, d))
    u = np.linalg.qr(z)[0]
    out = np.zeros((Y, B, d, d), dtype=complex)
    for j in range(d):
        out[:, j % B] += u[:, :, j, None] * u[:, None, :, j].conj()
    return out


def seesaw(pi, V, dim=2, iters=5, tol=1e-8, max_steps=500, seed=None, **solve_kw):
    """Best see-saw value over `iters` random restarts with local dimension `dim` (a lower bound on the quantum value).

    pi has shape (X, Y); V has shape (X, Y, A, B) and may hold any real payoffs.
    Extra keyword arguments go to cvxpy's Problem.solve. The default solver is Clarabel: cvxpy would otherwise pick
    SCS, whose ~1e-5 accuracy makes the see-saw value jitter so it never meets `tol` and runs all `max_steps`.
    """
    solve_kw.setdefault("solver", cp.CLARABEL)
    w = np.asarray(pi[:, :, None, None] * V, dtype=float)  # w[x, y, a, b]
    if w.shape[3] != 2 and w.shape[2] == 2:  # put the two-outcome player in Bob's seat, whose step is then exact
        w = w.transpose(1, 0, 3, 2)
    X, Y, A, B = w.shape
    rng = np.random.default_rng(seed)
    alice = _alice_sdp(X, A, dim)
    bob = _bob_two_outcome if B == 2 else _bob_sdp(Y, B, dim)

    best = -np.inf
    for _ in range(iters):
        bops = _random_projective(Y, B, dim, rng)
        prev = -np.inf
        for _ in range(max_steps):
            rho = alice(np.einsum("xyab,ybij->xaij", w, bops), **solve_kw)
            val, bops = bob(np.einsum("xyab,xaij->ybij", w, rho), **solve_kw)
            best = max(best, val)
            if abs(val - prev) <= tol:
                break
            prev = val
    return float(best)
