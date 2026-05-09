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

The outer iteration is a `lax.fori_loop` over a fixed-size pre-allocated
Krylov buffer, with the inner reorthogonalization a nested `lax.fori_loop`
over the populated prefix. Everything stays in jax-scalar arithmetic — no
`float()` casts force device-host syncs — so on GPU all `n_iters` matvecs
launch as one fused while-loop without Python dispatch in between.

Usage::

    from gato.solvers.lanczos import lanczos
    res = lanczos(H, psi0, n_iters=80, n_eigenstates=4)
    # res.eigenvalues[0] is the ground-state energy
    # res.eigenstates[..., 0] is ψ₀.
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
    converged: bool             # True if any beta fell below tolerance


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
    H : Hamiltonian (matrix-free; exposes H.apply and H.grid).
    psi0 : (Nx, Ny, Nz) starting vector; should have nonzero overlap with
        every target eigenstate.
    n_iters : Krylov dimension. The full `n_iters` steps are always run
        (no early-exit) so the inner loop can be staged once and executed
        on the device with no Python-side dispatch.
    n_eigenstates : how many Ritz pairs to return (lowest in the spectrum).
    tol : reported in the result if any β_j fell below this value, indicating
        Krylov saturation. Does not change the iteration count.
    """
    grid = H.grid
    shape = psi0.shape

    v0 = psi0 / jnp.sqrt(norm_sq(psi0, grid).real)
    real_dtype = jnp.real(v0).dtype

    V_buf = jnp.zeros((n_iters + 1,) + shape, dtype=v0.dtype).at[0].set(v0)
    alphas = jnp.zeros(n_iters, dtype=real_dtype)
    betas = jnp.zeros(n_iters, dtype=real_dtype)

    def body(j, carry):
        Vb, a_buf, b_buf = carry
        v_j = Vb[j]
        w = H.apply(v_j)
        # Subtract β_{j-1} v_{j-1}; at j=0 this is masked to zero.
        prev_idx = jnp.maximum(j - 1, 0)
        beta_prev = jnp.where(j > 0, b_buf[prev_idx], jnp.zeros((), dtype=real_dtype))
        w = w - beta_prev * Vb[prev_idx]
        # α_j = ⟨v_j, w⟩
        alpha = inner_product(v_j, w, grid).real
        a_buf = a_buf.at[j].set(alpha)
        w = w - alpha * v_j

        # Full reorthogonalization against Vb[0..j].
        def _reortho(k, w_inner):
            vk = Vb[k]
            coeff = inner_product(vk, w_inner, grid).real
            return w_inner - coeff * vk
        w = jax.lax.fori_loop(0, j + 1, _reortho, w)

        # β_j = ‖w‖; guard divison by zero to keep Vb finite even after
        # Krylov saturation. The corresponding v_{j+1} is left as zeros.
        beta = jnp.sqrt(norm_sq(w, grid).real)
        b_buf = b_buf.at[j].set(beta)
        safe_beta = jnp.where(beta > 0, beta, jnp.ones((), dtype=real_dtype))
        v_next = jnp.where(beta > 0, w / safe_beta, jnp.zeros_like(w))
        Vb = Vb.at[j + 1].set(v_next)
        return Vb, a_buf, b_buf

    V_buf, alphas, betas = jax.lax.fori_loop(
        0, n_iters, body, (V_buf, alphas, betas)
    )

    # Build the m×m symmetric tridiagonal T (m = n_iters). The j-th β couples
    # iterations j and j+1, so only the first n_iters-1 betas appear off-diagonal.
    T = jnp.diag(alphas)
    if n_iters >= 2:
        off = betas[: n_iters - 1]
        T = T + jnp.diag(off, k=1) + jnp.diag(off, k=-1)
    evals, evecs_T = jnp.linalg.eigh(T)

    # Reconstruct full-grid Ritz vectors from the Krylov basis. Use the
    # populated prefix V_buf[:n_iters]; V_buf[n_iters] is the next candidate
    # seed and not part of the Ritz expansion.
    V_stack = V_buf[:n_iters].reshape(n_iters, -1)
    psi_states = (evecs_T.T @ V_stack).reshape(n_iters, *shape)

    K = min(n_eigenstates, n_iters)
    eigenvalues = evals[:K]
    eigenstates = jnp.transpose(psi_states[:K], axes=(1, 2, 3, 0))

    converged = bool(jnp.any(betas[: max(n_iters - 1, 1)] < tol))

    return LanczosResult(
        eigenvalues=eigenvalues,
        eigenstates=eigenstates,
        n_iters=int(n_iters),
        converged=converged,
    )
