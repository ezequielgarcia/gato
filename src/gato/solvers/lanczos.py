"""Matrix-free Lanczos eigenvalue solver.

The Lanczos algorithm builds a Krylov subspace from a starting vector and the
action of a Hermitian operator. In that basis the operator is tridiagonal, and
its extreme eigenvalues converge exponentially fast (the gap-dependent power-
iteration rate). For a matrix-free Hamiltonian it gives access to the lowest
few eigenstates at the cost of a handful of `H.apply(psi)` evaluations per
iteration — far faster than imaginary-time for well-separated eigenvalues, and
with the bonus that it returns excited states directly.

We implement the variant with **full reorthogonalization**: every new Krylov
vector is projected against all previous ones. This is O(m² N³) for an m-step
run and is the numerically stable choice for m up to a few hundred. For
serious production use, periodic (partial) reorthogonalization is more
efficient; this is left for future work.

Usage::

    from gato.solvers.lanczos import lanczos
    evals, eigenstates = lanczos(H, psi0, n_iters=80, n_eigenstates=4)
    # evals[0] is the ground-state energy; eigenstates[..., 0] is ψ₀.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..grid import inner_product, norm_sq
from ..hamiltonian import Hamiltonian


@dataclass
class LanczosResult:
    eigenvalues: jax.Array      # shape (n_eigenstates,), sorted ascending
    eigenstates: jax.Array      # shape (*grid.shape, n_eigenstates)
    n_iters: int                # number of Lanczos steps actually taken
    converged: bool             # True if beta fell below tolerance


def lanczos(
    H: Hamiltonian,
    psi0: jax.Array,
    n_iters: int = 80,
    n_eigenstates: int = 4,
    tol: float = 1e-10,
) -> LanczosResult:
    """Compute the lowest `n_eigenstates` eigenpairs of H by Lanczos iteration.

    Parameters
    ----------
    H : Hamiltonian (matrix-free; exposes H.apply)
    psi0 : (Nx, Ny, Nz) starting vector; should have nonzero overlap with
        every target eigenstate (in practice, a generic random or physically
        motivated initial guess works).
    n_iters : maximum Krylov dimension. Typical: 50–200.
    n_eigenstates : how many eigenpairs to return (lowest in the spectrum).
    tol : early-exit tolerance on β (off-diagonal element); if β < tol the
        Krylov space has saturated and further iterations add no information.
    """
    grid = H.grid
    shape = psi0.shape

    # normalize the starting vector
    v = psi0 / jnp.sqrt(norm_sq(psi0, grid).real)

    V: list[jax.Array] = [v]              # Krylov basis vectors on the grid
    alpha_list: list[float] = []          # diagonal of the tridiagonal
    beta_list: list[float] = []           # off-diagonal

    converged = False
    for j in range(n_iters):
        w = H.apply(V[-1])
        if beta_list:
            w = w - beta_list[-1] * V[-2]

        alpha = float(inner_product(V[-1], w, grid).real)
        alpha_list.append(alpha)
        w = w - alpha * V[-1]

        # full reorthogonalization — protects against spurious eigenvalues
        for vk in V:
            w = w - inner_product(vk, w, grid) * vk

        beta = float(jnp.sqrt(norm_sq(w, grid).real))
        if beta < tol:
            converged = True
            break
        beta_list.append(beta)
        V.append(w / beta)

    m = len(alpha_list)
    # Build the m×m symmetric tridiagonal T. There are m alpha's and at most
    # m-1 useful beta's: a beta computed at the END of iteration j connects
    # iterations j and j+1, so it is only meaningful if we did iteration j+1.
    T = jnp.diag(jnp.asarray(alpha_list))
    if m >= 2:
        off = jnp.asarray(beta_list[: m - 1])
        T = T + jnp.diag(off, k=1) + jnp.diag(off, k=-1)
    evals, evecs_T = jnp.linalg.eigh(T)

    # Reconstruct full-grid eigenvectors from the Krylov basis. The loop
    # leaves V with m+1 entries when all m iterations complete (the extra v
    # is the next candidate seed); use only the first m for Ritz extraction.
    V_stack = jnp.stack([v.reshape(-1) for v in V[:m]], axis=0)  # (m, N³)
    psi_states = (evecs_T.T @ V_stack).reshape(m, *shape)       # (m, Nx, Ny, Nz)

    # pick the lowest K eigenpairs
    K = min(n_eigenstates, m)
    eigenvalues = evals[:K]
    eigenstates = jnp.transpose(psi_states[:K], axes=(1, 2, 3, 0))  # (..., K)

    return LanczosResult(
        eigenvalues=eigenvalues,
        eigenstates=eigenstates,
        n_iters=m,
        converged=converged,
    )
