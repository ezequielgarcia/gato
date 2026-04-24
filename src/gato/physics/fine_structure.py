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

from ..spectra import FINE_STRUCTURE
from .radial_hydrogen import (
    LogRadialGrid,
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
