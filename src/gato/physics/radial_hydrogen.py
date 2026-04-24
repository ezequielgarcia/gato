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


def _pad_boundary(
    u: jax.Array, width: int = 2, left_sign: int = -1
) -> jax.Array:
    """Pad u with signed reflection at ξ=0 and zeros at ξ=1.

    The physical boundary at r=0 depends on angular momentum: the radial
    function u(r) ≈ r^(ℓ+1) near the origin, so under ξ ↔ -ξ (equivalently
    r ↔ -r through 0) it has parity (-1)^(ℓ+1). Concretely:

        left_sign = -1  for ℓ = 0, 2, 4, ...   (u odd through r=0)
        left_sign = +1  for ℓ = 1, 3, 5, ...   (u even through r=0)

    Using the wrong parity biases the kinetic energy by O(h²) and spoils
    the 4th-order convergence; zero-padding works for ℓ ≥ 1 too but is
    noticeably worse than the correct reflection.
    """
    left = left_sign * jnp.flip(u[:width])
    right = jnp.zeros(width, dtype=u.dtype)
    return jnp.concatenate([left, u, right])


def _left_sign_for_ell(ell: int) -> int:
    return -1 if (ell % 2 == 0) else +1


def _d_dxi_4th(u: jax.Array, h: float, left_sign: int = -1) -> jax.Array:
    p = _pad_boundary(u, 2, left_sign)
    return (p[:-4] - 8 * p[1:-3] + 8 * p[3:-1] - p[4:]) / (12 * h)


def _d2_dxi2_4th(u: jax.Array, h: float, left_sign: int = -1) -> jax.Array:
    p = _pad_boundary(u, 2, left_sign)
    return (
        -p[:-4] + 16 * p[1:-3] - 30 * p[2:-2] + 16 * p[3:-1] - p[4:]
    ) / (12 * h * h)


def radial_laplacian(
    u: jax.Array, grid: LogRadialGrid, ell: int = 0
) -> jax.Array:
    """d²u/dr² on the log grid via chain rule + 4th-order ξ-stencils.

    The ``ell`` argument only selects the correct parity at the r=0 boundary;
    the centrifugal barrier itself is added in `radial_hamiltonian`.
    """
    h = grid.h_xi
    sign = _left_sign_for_ell(ell)
    du = _d_dxi_4th(u, h, sign)
    d2u = _d2_dxi2_4th(u, h, sign)
    rp = grid.r_prime()
    rpp = grid.r_double_prime()
    return d2u / (rp * rp) - rpp / (rp ** 3) * du


def radial_kinetic(
    u: jax.Array, grid: LogRadialGrid, ell: int = 0
) -> jax.Array:
    return -0.5 * radial_laplacian(u, grid, ell)


def coulomb_potential(grid: LogRadialGrid, Z: float = 1.0) -> jax.Array:
    """V(r) = -Z/r. No softening: the log grid never samples r = 0."""
    return -Z / grid.r()


def centrifugal_barrier(grid: LogRadialGrid, ell: int) -> jax.Array:
    """Angular-momentum barrier ℓ(ℓ+1)/(2 r²) on the log grid.

    This is the only new term needed to extend the ℓ=0 radial equation to
    arbitrary ℓ: the reduction ψ_{nℓm}(r,θ,φ) = (u_{nℓ}(r)/r) Y_{ℓm}(θ,φ)
    turns −½∇² into −½∂²/∂r² + ℓ(ℓ+1)/(2r²) acting on u. Returns zero for
    ℓ = 0.
    """
    if ell == 0:
        return jnp.zeros_like(grid.r())
    r = grid.r()
    return 0.5 * ell * (ell + 1) / (r * r)


def radial_hamiltonian(
    u: jax.Array, grid: LogRadialGrid, Z: float = 1.0, ell: int = 0
) -> jax.Array:
    V = coulomb_potential(grid, Z) + centrifugal_barrier(grid, ell)
    return radial_kinetic(u, grid, ell) + V * u


def radial_inner_product(
    u: jax.Array, v: jax.Array, grid: LogRadialGrid
) -> jax.Array:
    """⟨u | v⟩ = ∫₀^∞ u v dr ≈ h_ξ Σ_j r'(ξ_j) u_j v_j."""
    return grid.h_xi * jnp.sum(u * v * grid.r_prime())


def build_hamiltonian_matrix(
    grid: LogRadialGrid, Z: float = 1.0, ell: int = 0
) -> jax.Array:
    """Dense (N, N) matrix H such that (Hu)_j = Σ_k H_{jk} u_k.

    Not symmetric: the operator is Hermitian under the weighted inner product
    ⟨u, v⟩ = h_ξ Σ r'_j u_j v_j, which means W·H is symmetric where
    W = diag(r'). See ``solve_ground_state`` for the symmetrization.
    """
    N = grid.N

    def column(i):
        e_i = jnp.zeros(N).at[i].set(1.0)
        return radial_hamiltonian(e_i, grid, Z, ell)

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
    energies, states = solve_bound_states(grid, Z, K=1, ell=0)
    return energies[0], states[:, 0]


def solve_bound_states(
    grid: LogRadialGrid, Z: float = 1.0, K: int = 5, ell: int = 0
) -> tuple[jax.Array, jax.Array]:
    """Lowest K bound states (E_n, u_n) in the ℓ-channel via dense diagonalization.

    Returns
    -------
    energies : shape (K,), ascending.
    states : shape (N, K), each column u_{nℓ}(r) normalized under
        ⟨u, u⟩_W = h_ξ Σ r'_j u_j². The full wavefunction is
        ψ_{nℓm}(r,θ,φ) = (u_{nℓ}(r)/r) Y_{ℓm}(θ,φ).

    For hydrogen (pure Coulomb, Z) the spectrum is E_n = −Z²/(2n²) with
    n ≥ ℓ + 1. The ℓ-degeneracy is a Coulomb accident (not a generic
    spherical-potential feature), so e.g. 2s (ℓ=0) and 2p (ℓ=1) both sit
    at −Z²/8. Extending beyond ℓ=0 unlocks real electric-dipole matrix
    elements between solver-produced states: the Δℓ = ±1 selection rule
    is an *angular* statement (Wigner–Eckart on cos θ between Y_{ℓm}'s),
    and the radial integral ⟨u_{n'ℓ'} | r | u_{nℓ}⟩ is a one-liner on
    this grid.
    """
    H = build_hamiltonian_matrix(grid, Z, ell)
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


def radial_dipole(
    u_final: jax.Array, u_initial: jax.Array, grid: LogRadialGrid
) -> jax.Array:
    """Radial piece ⟨u_{n'ℓ'} | r | u_{nℓ}⟩ = ∫₀^∞ u' · r · u dr.

    This is the ℓ-independent part of any electric-dipole matrix element
    between two spherical states. The full ⟨ψ_f | z | ψ_i⟩ (or x, y) is

        ⟨ψ_f | ẑ_α | ψ_i⟩ = ⟨u_f | r | u_i⟩  ·  ⟨Y_{ℓ'm'} | r̂_α | Y_{ℓm}⟩,

    where the angular factor — a standard Gaunt coefficient — enforces
    Δℓ = ±1 and Δm ∈ {−1, 0, +1}. Example: for 1s → 2p_z,
        angular piece ⟨Y_{10} | cosθ | Y_{00}⟩ = 1/√3
        radial piece ⟨u_{2p} | r | u_{1s}⟩ = 256/(81√6)
        product = 128√2/243 ≈ 0.7449 a₀ (Bethe–Salpeter §63).
    """
    return grid.h_xi * jnp.sum(u_final * grid.r() * u_initial * grid.r_prime())
