"""Log-spaced 1D radial solver for single-center hydrogenic atoms.

A standalone cross-check for Phase 1. This module does **not** participate in
the 3D Cartesian stack — it's here so the hydrogen ground state can be
recovered with a *pure* -Z/r potential (no softening) to independently validate
the ε → 0 linear extrapolation reported in the main benchmark.

Physics
-------
For a spherically-symmetric ℓ=0 state write ψ(r) = u(r) / r. The
time-independent Schrödinger equation then reduces to the 1D radial equation

    -½ u''(r) - (Z/r) u(r) = E u(r),    u(0) = u(r_max) = 0,
    ∫₀^∞ |u|² dr = 1.

Hydrogenic ground state: u(r) = 2 Z^{3/2} r e^{-Zr}, E₀ = -Z²/2.

Discretization
--------------
Uniform cell-centered ξ ∈ (0, 1) with N points, and log mapping

    r(ξ) = r_min · (exp(α ξ) - 1),    α = log(r_max / r_min + 1).

Then r(0) = 0 exactly, r(1) = r_max, r'(ξ) = α (r + r_min), r'' = α r'.
Near ξ=0 the mapping is linear (r ≈ α r_min ξ); at large ξ it is exponential.
The scale parameter r_min controls where the crossover happens, *not* the
smallest grid r-value (which is r_min·(exp(α h_ξ/2)−1), typically much smaller).

4th-order central-difference stencils act on u(ξ) and the r-space derivatives
come from the chain rule

    u''(r) = u''_ξ / r'² - r''/r'³ · u'_ξ.

Boundaries
----------
At ξ=0: u(0)=0 is implied by ψ=u/r being finite. The physically correct
padding is *odd extension*, u_{-1-j} = -u_j, which matches the odd parity
of u(r) through r=0 and restores the full 4th-order stencil accuracy at
the first two interior points. (Zero-padding there biases the kinetic
energy by O(1e-4) and spoils the grid convergence.)

At ξ=1: zero-padding is correct since u is exponentially small at r_max.

Scope: ℓ=0 only, single nucleus at r=0. Not a replacement for the 3D
grid solver in any way — molecules, non-spherical states, and open-shell
atoms all require the Cartesian stack.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class LogRadialGrid:
    """Cell-centered log-spaced 1D grid in (r_min, r_max)."""

    N: int
    r_min: float
    r_max: float

    @property
    def alpha(self) -> float:
        return math.log(self.r_max / self.r_min + 1.0)

    @property
    def h_xi(self) -> float:
        return 1.0 / self.N

    def xi(self) -> jax.Array:
        return (jnp.arange(self.N) + 0.5) * self.h_xi

    def r(self) -> jax.Array:
        return self.r_min * (jnp.exp(self.alpha * self.xi()) - 1.0)

    def r_prime(self) -> jax.Array:
        return self.alpha * (self.r() + self.r_min)

    def r_double_prime(self) -> jax.Array:
        return self.alpha * self.r_prime()


def _pad_odd_zero(u: jax.Array, width: int = 2) -> jax.Array:
    """Pad u with odd reflection at ξ=0 and zeros at ξ=1.

    Odd reflection (u_{-1-j} = -u_j) is the physical boundary for u(r) at r=0:
    the radial function vanishes at the origin and has odd parity through it,
    so the stencil sees a smoothly extended function instead of an artificial
    jump to zero.
    """
    left = -jnp.flip(u[:width])
    right = jnp.zeros(width, dtype=u.dtype)
    return jnp.concatenate([left, u, right])


def _d_dxi_4th(u: jax.Array, h: float) -> jax.Array:
    p = _pad_odd_zero(u, 2)
    return (p[:-4] - 8 * p[1:-3] + 8 * p[3:-1] - p[4:]) / (12 * h)


def _d2_dxi2_4th(u: jax.Array, h: float) -> jax.Array:
    p = _pad_odd_zero(u, 2)
    return (
        -p[:-4] + 16 * p[1:-3] - 30 * p[2:-2] + 16 * p[3:-1] - p[4:]
    ) / (12 * h * h)


def radial_laplacian(u: jax.Array, grid: LogRadialGrid) -> jax.Array:
    """d²u/dr² on the log grid via chain rule + 4th-order ξ-stencils."""
    h = grid.h_xi
    du = _d_dxi_4th(u, h)
    d2u = _d2_dxi2_4th(u, h)
    rp = grid.r_prime()
    rpp = grid.r_double_prime()
    return d2u / (rp * rp) - rpp / (rp ** 3) * du


def radial_kinetic(u: jax.Array, grid: LogRadialGrid) -> jax.Array:
    return -0.5 * radial_laplacian(u, grid)


def coulomb_potential(grid: LogRadialGrid, Z: float = 1.0) -> jax.Array:
    """V(r) = -Z/r. No softening: the log grid never samples r = 0."""
    return -Z / grid.r()


def radial_hamiltonian(u: jax.Array, grid: LogRadialGrid, Z: float = 1.0) -> jax.Array:
    return radial_kinetic(u, grid) + coulomb_potential(grid, Z) * u


def radial_inner_product(
    u: jax.Array, v: jax.Array, grid: LogRadialGrid
) -> jax.Array:
    """⟨u | v⟩ = ∫₀^∞ u v dr ≈ h_ξ Σ_j r'(ξ_j) u_j v_j."""
    return grid.h_xi * jnp.sum(u * v * grid.r_prime())


def build_hamiltonian_matrix(grid: LogRadialGrid, Z: float = 1.0) -> jax.Array:
    """Dense (N, N) matrix H such that (Hu)_j = Σ_k H_{jk} u_k.

    Not symmetric: the operator is Hermitian under the weighted inner product
    ⟨u, v⟩ = h_ξ Σ r'_j u_j v_j, which means W·H is symmetric where
    W = diag(r'). See ``solve_ground_state`` for the symmetrization.
    """
    N = grid.N

    def column(i):
        e_i = jnp.zeros(N).at[i].set(1.0)
        return radial_hamiltonian(e_i, grid, Z)

    return jax.vmap(column)(jnp.arange(N)).T


def solve_ground_state(
    grid: LogRadialGrid, Z: float = 1.0
) -> tuple[jax.Array, jax.Array]:
    """Ground-state (E₀, u₀) via dense symmetric diagonalization.

    Derivation of the symmetrization: H is Hermitian under ⟨·,·⟩_W with
    W = diag(r'). Equivalently, W·H is symmetric. Conjugating by W^{1/2}
    gives the standard-symmetric matrix H̃ = W^{1/2} H W^{-1/2} with the
    same spectrum as H. Eigenvectors of H̃ map back via u = W^{-1/2} ṽ.
    """
    energies, states = solve_bound_states(grid, Z, K=1)
    return energies[0], states[:, 0]


def solve_bound_states(
    grid: LogRadialGrid, Z: float = 1.0, K: int = 5
) -> tuple[jax.Array, jax.Array]:
    """Lowest K ℓ=0 bound states (E_n, u_n) via dense diagonalization.

    Returns
    -------
    energies : shape (K,), ascending.
    states : shape (N, K), each column u_n normalized under
        ⟨u, u⟩_W = h_ξ Σ r'_j u_j².

    For hydrogen (Z, ℓ=0) the spectrum is E_n = -Z²/(2n²), n = 1, 2, ...,
    giving the Lyman/Balmer/Paschen line positions directly as eigenvalue
    differences. Selection rules require Δℓ = ±1 for electric-dipole
    transitions, so line *positions* are visible in ℓ=0 alone (the
    spectrum is ℓ-degenerate for pure Coulomb) but line *strengths*
    between specific (n,ℓ) → (n',ℓ') pairs need ℓ>0 channels.
    """
    H = build_hamiltonian_matrix(grid, Z)
    w_sqrt = jnp.sqrt(grid.r_prime())
    w_inv_sqrt = 1.0 / w_sqrt
    H_sym = (w_sqrt[:, None] * H) * w_inv_sqrt[None, :]
    H_sym = 0.5 * (H_sym + H_sym.T)
    eigvals, eigvecs = jnp.linalg.eigh(H_sym)
    energies = eigvals[:K]
    states = eigvecs[:, :K] * w_inv_sqrt[:, None]
    norms = jnp.sqrt(
        grid.h_xi * jnp.sum(states * states * grid.r_prime()[:, None], axis=0)
    )
    states = states / norms[None, :]
    return energies, states
