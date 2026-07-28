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


def default_krylov_dim(N: int, n_eigenstates: int = 1) -> int:
    """Krylov dimension that resolves the lowest states on an N³ grid.

    The number of Lanczos steps needed to converge the bottom of the spectrum
    scales as $\\sqrt{W/g}$, where $W$ is the operator's spectral width and $g$
    the gap above the states of interest. For a finite-difference kinetic
    operator $W \\sim h^{-2}$ while $g$ is set by the physics and is
    $h$-independent, so the requirement grows like $1/h \\propto N$ — a *fixed*
    Krylov dimension silently degrades as the grid is refined.

    The 1.25·N coefficient is calibrated empirically on H₂O + HGH (8 valence
    electrons, L = 14 a₀), where the pass/fail threshold is sharp: N = 64
    needs ≳ 80 steps (60 diverges), N = 96 needs ≳ 120 (100 diverges). Below
    that threshold the highest occupied state comes out qualitatively wrong
    (e.g. -0.11 Ha instead of -0.52 Ha at N = 96) and the SCF never converges,
    so the failure is loud rather than a small loss of precision.

    The $24 n + 24$ floor covers the part of the requirement that does *not*
    come from the spectral width. Pulling apart a *degenerate* pair from a
    single Lanczos starting vector is slow no matter how coarse the grid, and
    it is the binding constraint for HCl: at L = 14 a₀ its Cl 3π pair needs
    ≳ 120 steps at both N = 40 and N = 48, where the 1/h term alone would ask
    for only 50–60. Under-resolved, the pair splits spuriously (-0.498 /
    -0.497 Ha instead of degenerate) and the SCF stalls at max_iters.

    **This rule is calibrated for non-degenerate occupied manifolds and is NOT
    sufficient when a molecule has a symmetry-required degeneracy.** The two
    effects above are combined with `max`, which assumes the larger dominates;
    for HCl that assumption fails at finer grids. Measured on HCl at L = 14 a₀:
    N = 64 is resolved by the 120 this returns, but N = 80 needs ≳ 200 and gets
    120, which silently yields a wrong answer (the 3π pair does not appear at
    all and the fourth orbital lands at -0.16 Ha instead of -0.48 Ha, with the
    total energy off by 0.75 Ha while `converged` still reports True).

    The root cause is structural, not a bad constant: a single Lanczos starting
    vector spans one direction per invariant Krylov subspace, so an exactly
    degenerate level can only be split by round-off. No choice of scalar
    formula fixes that in general — the real fix is a block method (block
    Lanczos / LOBPCG) seeded with n_eigenstates vectors. Until then, molecules
    in point groups with only 1-D irreps (C1, Cs, C2, C2v, C2h, D2h — H₂O among
    them) are safe; ones with π, e or t degeneracies (HCl, HF, N₂, CO, NH₃,
    CH₄) need `lanczos_iters` set explicitly and verified by hand.

    Note that the **Ritz residual does not detect this failure**, so don't
    reach for it as a safety net — it was tried and rejected. The residual
    |β_m · s_k[m−1]| measures whether the returned pairs are *converged*
    eigenpairs, not whether they are the *wanted* ones. When the Krylov space
    never contains the second component of a degenerate pair, Lanczos converges
    tightly onto a different, complete-looking set of states. Measured on HCl at
    N = 80: the wrong answer (m = 120) has residual 2.2e-5 while the right one
    (m = 200) has 3.1e-5 — the bad run scores *better*. Correct water runs at
    N = 64 and N = 96 spread over 1.7e-6..5.9e-4, so no threshold separates
    them. A residual cannot see a *skipped* eigenvalue; only a block method or
    an explicit convergence study in m can.

    Cost scales as O(m²N³) in time and O(mN³) in memory, so this is not a
    quantity to over-provision on a large grid.
    """
    return max(24 * n_eigenstates + 24, (5 * N) // 4)


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

    K = min(n_eigenstates, n_iters)

    # Reconstruct full-grid Ritz vectors from the Krylov basis. Use the
    # populated prefix V_buf[:n_iters]; V_buf[n_iters] is the next candidate
    # seed and not part of the Ritz expansion.
    #
    # Only the K lowest Ritz pairs are ever returned, so restrict the
    # coefficient matrix to those columns *before* the matmul. Reconstructing
    # all n_iters states first would allocate an (n_iters, N³) array and do
    # n_iters/K times more work — at n_iters=80 that array is the peak memory
    # of the whole solver and caps the affordable grid size.
    V_stack = V_buf[:n_iters].reshape(n_iters, -1)
    psi_states = (evecs_T[:, :K].T @ V_stack).reshape(K, *shape)

    eigenvalues = evals[:K]
    eigenstates = jnp.transpose(psi_states, axes=(1, 2, 3, 0))

    converged = bool(jnp.any(betas[: max(n_iters - 1, 1)] < tol))

    return LanczosResult(
        eigenvalues=eigenvalues,
        eigenstates=eigenstates,
        n_iters=int(n_iters),
        converged=converged,
    )
