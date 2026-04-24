"""Phase 7 — perturbative scalar-relativistic corrections (mass–velocity + Darwin).

The leading relativistic shifts to a nonrelativistic Schrödinger eigenvalue
are the mass–velocity and Darwin terms (spin–orbit vanishes for ℓ=0 and is
absorbed into the scalar-relativistic story we don't model here):

    H_MV = -p^4 / (8 c²)                            (all ℓ)
    H_D  = (π Z / (2 c²)) δ³(r − R)                 (contact term, ℓ=0 only)

In atomic units c = 1/α, so α² = 1/c² and both operators scale as α².
Evaluating them as *expectation values* on a nonrelativistic ground state
is cheap (one extra integral per state) and gives the same first-order
energy shift as ZORA expanded to O(α²). This module implements exactly
that — the "alternative perturbative mass-velocity + Darwin, as a
cross-check against ZORA" item from README §5.7.

For hydrogen 1s (analytic):

    ⟨p⁴⟩_{1s} = 5 Z⁴          →  ⟨H_MV⟩ = −5 α² Z⁴ / 8
    |ψ(0)|²   = Z³ / π        →  ⟨H_D⟩  = + α² Z⁴ / 2
    total                      =  −α² Z⁴ / 8

which is the Sommerfeld fine-structure shift for the (n=1, j=1/2) level.
The Darwin term relies on ℓ=0 for its δ³(r) contact value; ℓ>0 orbitals
have ψ(0) = 0 and the contact term drops.

Scope: ℓ=0 radial states on the Phase 1 log-radial grid. ZORA (the full
operator) is a separate slice on top of this.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ..spectra import FINE_STRUCTURE, SPEED_OF_LIGHT_AU
from .radial_hydrogen import (
    LogRadialGrid,
    _d_dxi_4th,
    _left_sign_for_ell,
    coulomb_potential,
    radial_inner_product,
    radial_laplacian,
)


def p4_expectation(u: jax.Array, grid: LogRadialGrid) -> jax.Array:
    """⟨p⁴⟩ for a normalized ℓ=0 radial state u(r).

    Derivation: for ψ = (u/r) Y_{00},

        p² ψ = −∇²ψ = −u''(r) / r        (ℓ=0 identity),
        ⟨p⁴⟩ = ⟨p²ψ | p²ψ⟩ = ∫₀^∞ (u''(r))² dr.

    The 4th-order chain-rule `radial_laplacian` produces u''(r) directly,
    and `radial_inner_product` integrates over r with the log-grid weight.
    """
    u_rr = radial_laplacian(u, grid, ell=0)
    return radial_inner_product(u_rr, u_rr, grid)


def mass_velocity_expectation(u: jax.Array, grid: LogRadialGrid) -> jax.Array:
    """⟨H_MV⟩ = −α² / 8 · ⟨p⁴⟩ for a normalized ℓ=0 radial state."""
    return -0.5 * 0.25 * (FINE_STRUCTURE ** 2) * p4_expectation(u, grid)


def psi_at_origin_sq(u: jax.Array, grid: LogRadialGrid) -> jax.Array:
    """|ψ(0)|² for a ℓ=0 state, extrapolated from the innermost grid point.

    For ψ = (u/r) Y_{00}, |ψ(0)|² = R(0)² / (4π) with R(r) = u(r)/r. On
    the log-radial grid the smallest r ≈ r_min · α · h_ξ / 2 is many
    orders of magnitude smaller than physical length scales, so using
    u[0] / r[0] as R(0) is correct to O((r[0])²) for any smooth
    exponentially-bound state.
    """
    r = grid.r()
    R0 = u[0] / r[0]
    return (R0 ** 2) / (4.0 * jnp.pi)


def darwin_expectation(
    u: jax.Array, grid: LogRadialGrid, Z: float
) -> jax.Array:
    """⟨H_D⟩ = (π Z α² / 2) |ψ(0)|² for a ℓ=0 state.

    The contact form applies only to ℓ=0 (all higher ℓ have ψ(0) = 0 and
    the Darwin correction vanishes identically).
    """
    return 0.5 * jnp.pi * Z * (FINE_STRUCTURE ** 2) * psi_at_origin_sq(u, grid)


def fine_structure_shift(
    u: jax.Array, grid: LogRadialGrid, Z: float
) -> jax.Array:
    """Total leading-α² relativistic shift ⟨H_MV⟩ + ⟨H_D⟩ for ℓ=0.

    For hydrogen 1s this reduces to the closed-form −α² Z⁴ / 8, matching
    the Sommerfeld fine-structure energy for (n=1, j=1/2) without any
    spin-orbit piece (which vanishes for ℓ=0 anyway).
    """
    return mass_velocity_expectation(u, grid) + darwin_expectation(u, grid, Z)


# -------------------- ZORA (full scalar-relativistic operator) --------------------

def zora_K(grid: LogRadialGrid, Z: float) -> jax.Array:
    """ZORA kinetic weight K(r) = c² / (2c² − V(r)) for V = −Z/r.

    In the nonrelativistic limit V ≪ c², K → 1/2 and T_ZORA → −½ ∂²/∂r²
    (standard kinetic). Near the nucleus V → −∞, K → 0, which softens the
    kinetic energy exactly where the relativistic mass-enhancement should
    reduce it — that's the physical content of ZORA.
    """
    c2 = SPEED_OF_LIGHT_AU ** 2
    V = -Z / grid.r()
    return c2 / (2.0 * c2 - V)


def kinetic_zora_radial(
    u: jax.Array, grid: LogRadialGrid, Z: float
) -> jax.Array:
    """ZORA kinetic on the ℓ=0 log-radial grid.

    Derivation. In 3D T_ZORA ψ = −∇·(K ∇ψ) with K(r) = c²/(2c² − V(r)).
    For spherically symmetric ψ = u(r)/r (the ℓ=0 reduction) the divergence
    unfolds as

        −∇·(K ∇ψ) = −K u''/r − K' u'/r + (K'/r²) u.

    Multiplying the eigenequation by r (so u, not ψ, is the dependent
    variable) gives the operator that acts on u:

        T_ZORA u = −K(r) u''(r) − K'(r) u'(r) + (K'(r)/r) u(r).

    The (K'/r) u term is what's *missing* from a naïve 1D ZORA translation
    and is numerically dominant: without it the discretized eigenvalue
    shift exceeds the physical Sommerfeld shift by two orders of magnitude
    and diverges with N. With it included, the continuum operator is
    self-adjoint (∫ K u' v' dr + ∫ (K'/r) u v dr is symmetric in u, v),
    and the discretized matrix is symmetrized once more in
    `solve_zora_ground_state` to absorb any residual O(h⁴) asymmetry.

    K(r) and K'(r) are closed-form for V = −Z/r; only u'(r) and u''(r)
    come from the 4th-order chain-rule stencils.
    """
    c2 = SPEED_OF_LIGHT_AU ** 2
    r = grid.r()
    V = -Z / r
    two_c2_minus_V = 2.0 * c2 - V
    K = c2 / two_c2_minus_V
    # dV/dr = Z/r² for V = −Z/r, so dK/dr = c² · (dV/dr) / (2c² − V)².
    dKdr = c2 * (Z / (r * r)) / (two_c2_minus_V ** 2)

    sign = _left_sign_for_ell(0)
    du_dxi = _d_dxi_4th(u, grid.h_xi, sign)
    u_r = du_dxi / grid.r_prime()
    u_rr = radial_laplacian(u, grid, ell=0)
    return -K * u_rr - dKdr * u_r + (dKdr / r) * u


def radial_hamiltonian_zora(
    u: jax.Array, grid: LogRadialGrid, Z: float
) -> jax.Array:
    """ℓ=0 ZORA Hamiltonian H_ZORA u = T_ZORA u − (Z/r) u."""
    return kinetic_zora_radial(u, grid, Z) + coulomb_potential(grid, Z) * u


def build_zora_hamiltonian_matrix(
    grid: LogRadialGrid, Z: float
) -> jax.Array:
    """Dense (N, N) matrix form of the ZORA radial Hamiltonian, ℓ=0."""
    N = grid.N

    def column(i):
        e_i = jnp.zeros(N).at[i].set(1.0)
        return radial_hamiltonian_zora(e_i, grid, Z)

    return jax.vmap(column)(jnp.arange(N)).T


def solve_zora_ground_state(
    grid: LogRadialGrid, Z: float = 1.0
) -> tuple[jax.Array, jax.Array]:
    """Ground-state (E_ZORA, u_ZORA) of the ℓ=0 ZORA Hamiltonian.

    Same W-symmetrization as `solve_ground_state`: H is Hermitian under
    ⟨·, ·⟩_W with W = diag(r'), so W^{1/2} H W^{-1/2} is symmetric in the
    standard Euclidean sense and `jnp.linalg.eigh` applies cleanly. The
    returned u is normalized under the radial inner product so that
    ⟨u_ZORA | u_ZORA⟩ = 1.

    For hydrogen, E_ZORA − E_NR reproduces the Sommerfeld shift −α² Z⁴ / 8
    at O(α²); higher-order deviations enter at O(α⁴) and are ~1 % of the
    leading shift for Z = 1.
    """
    H = build_zora_hamiltonian_matrix(grid, Z)
    w_sqrt = jnp.sqrt(grid.r_prime())
    w_inv_sqrt = 1.0 / w_sqrt
    H_sym = (w_sqrt[:, None] * H) * w_inv_sqrt[None, :]
    H_sym = 0.5 * (H_sym + H_sym.T)
    eigvals, eigvecs = jnp.linalg.eigh(H_sym)
    E0 = eigvals[0]
    u0 = eigvecs[:, 0] * w_inv_sqrt
    u0 = u0 / jnp.sqrt(radial_inner_product(u0, u0, grid))
    return E0, u0
