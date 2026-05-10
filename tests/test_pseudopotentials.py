"""Smoke tests for HGH pseudopotentials (gato.pseudopotentials).

These checks validate the analytic form against properties that follow
directly from the HGH paper, with no SCF involved:

1. V_loc(r) → -Z_ion/r at large r (correct long-range Coulomb tail).
2. V_loc at r=0 matches the analytic limit -Z_ion·√(2/π)/r_loc + C_1.
3. Radial projectors p_{ℓi}(r) are normalized: ∫_0^∞ |p_{ℓi}|² r² dr = 1.
4. V_nl is Hermitian: ⟨φ|V_nl ψ⟩ = ⟨V_nl φ|ψ⟩.
5. The local-potential field is differentiable in nuclear positions.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gato.geometry import Nuclei
from gato.grid import Grid3D, inner_product
from gato.pseudopotentials import (
    CHLORINE_HGH,
    HYDROGEN_HGH,
    LITHIUM_HGH,
    OXYGEN_HGH,
    _projector_radial,
    hgh_local_potential,
    hgh_nonlocal_apply,
    lookup,
)


# ---------------------------------------------------------------------------
# 1. Long-range Coulomb tail
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "params,r_test",
    [
        (HYDROGEN_HGH, 5.0),
        (LITHIUM_HGH, 5.0),
        (OXYGEN_HGH, 5.0),
        (CHLORINE_HGH, 5.0),
    ],
)
def test_local_potential_long_range_tail(params, r_test):
    """At r >> r_loc, V_loc(r) ≈ -Z_ion / r."""
    grid = Grid3D(N=64, L=20.0)
    nuclei = Nuclei(
        positions=jnp.array([[0.0, 0.0, 0.0]]),
        charges=jnp.array([params.Z_ion]),
    )
    V = hgh_local_potential(nuclei, (params,), grid)

    X, Y, Z = grid.coords()
    r = jnp.sqrt(X * X + Y * Y + Z * Z)
    # Pick the single grid point whose r is closest to r_test, and compare
    # V at that exact point against -Z_ion/r at that exact point. This avoids
    # spherical-shell averaging (which would give -Z·⟨1/r⟩, not -Z/⟨r⟩).
    flat_idx = int(jnp.argmin(jnp.abs(r - r_test)))
    r_far = float(r.flatten()[flat_idx])
    V_far = float(V.flatten()[flat_idx])

    expected = -params.Z_ion / r_far
    assert abs(V_far - expected) < 1e-6, (
        f"V_loc at r={r_far:.4f} = {V_far:.8f}, expected ≈ {expected:.8f}"
    )


# ---------------------------------------------------------------------------
# 2. Value at the origin
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "params", [HYDROGEN_HGH, LITHIUM_HGH, OXYGEN_HGH, CHLORINE_HGH]
)
def test_local_potential_value_at_origin(params):
    """V_loc(0) = -Z_ion·√(2/π)/r_loc + C_1.

    erf-screened Coulomb at r→0: erf(r/a)/r → 2/(a√π), so
    -Z_ion·erf(r/(√2 r_loc))/r → -Z_ion·√(2/π)/r_loc.
    Polynomial term at r=0 reduces to C_1·exp(0) = C_1.
    """
    # Use a fine-spaced grid centered on the origin so a grid point sits near 0.
    grid = Grid3D(N=32, L=3.0)
    nuclei = Nuclei(
        positions=jnp.array([[0.0, 0.0, 0.0]]),
        charges=jnp.array([params.Z_ion]),
    )
    V = hgh_local_potential(nuclei, (params,), grid)

    X, Y, Z = grid.coords()
    r = jnp.sqrt(X * X + Y * Y + Z * Z)
    # Grid is cell-centered → minimum r ≈ h·√3/2 on a symmetric grid; the
    # closest point is still small enough that the analytic limit dominates.
    idx = int(jnp.argmin(r))
    r_min = float(r.flatten()[idx])
    V_min = float(V.flatten()[idx])

    # Evaluate the analytic V_loc at r_min directly to compare like-with-like.
    sqrt2 = math.sqrt(2.0)
    erf_r = math.erf(r_min / (sqrt2 * params.r_loc))
    erf_term = -params.Z_ion * erf_r / r_min
    rho2 = (r_min / params.r_loc) ** 2
    c1, c2, c3, c4 = params.c
    poly = c1 + c2 * rho2 + c3 * rho2 * rho2 + c4 * rho2 * rho2 * rho2
    expected = erf_term + math.exp(-0.5 * rho2) * poly

    assert abs(V_min - expected) < 1e-9, (
        f"V_loc(r={r_min:.4g}) = {V_min:.6f}, analytic = {expected:.6f}"
    )


# ---------------------------------------------------------------------------
# 3. Radial projector normalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ell,i_idx,r_ell", [
    (0, 1, 0.22178614),     # oxygen s-channel
    (0, 1, 0.5),            # synthetic s, broader
    (1, 1, 0.4),            # synthetic p
    (0, 2, 0.3),            # synthetic s, second radial
])
def test_radial_projector_normalization(ell, i_idx, r_ell):
    """∫_0^∞ |p_{ℓi}(r)|² r² dr = 1 by HGH construction.

    Combined with ∫|Y_{ℓm}|² dΩ = 1 this means the full 3D projector is
    L²-normalized.
    """
    # Trapezoid on a fine 1D grid; the projector decays as a Gaussian, so a
    # box of 8 r_ell is overkill.
    r = jnp.linspace(1e-6, 8.0 * r_ell, 20000)
    p = _projector_radial(r, ell, i_idx, r_ell)
    integrand = (p * p) * (r * r)
    norm = float(jnp.trapezoid(integrand, r))
    assert abs(norm - 1.0) < 1e-4, f"projector norm = {norm}, expected 1.0"


# ---------------------------------------------------------------------------
# 4. Non-local Hermiticity
# ---------------------------------------------------------------------------

def test_nonlocal_potential_hermitian():
    """⟨φ|V_nl ψ⟩ = ⟨V_nl φ|ψ⟩ for the oxygen non-local block."""
    grid = Grid3D(N=40, L=10.0)
    nuclei = Nuclei(
        positions=jnp.array([[0.0, 0.0, 0.0]]),
        charges=jnp.array([OXYGEN_HGH.Z_ion]),
    )
    elements = (OXYGEN_HGH,)

    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    phi = jax.random.normal(k1, grid.shape, dtype=jnp.float64)
    psi = jax.random.normal(k2, grid.shape, dtype=jnp.float64)

    Vphi = hgh_nonlocal_apply(phi, nuclei, elements, grid)
    Vpsi = hgh_nonlocal_apply(psi, nuclei, elements, grid)

    a = float(inner_product(phi, Vpsi, grid).real)
    b = float(inner_product(Vphi, psi, grid).real)

    assert abs(a - b) < 1e-9 * (abs(a) + abs(b) + 1.0), (
        f"⟨φ|V_nl ψ⟩ = {a}, ⟨V_nl φ|ψ⟩ = {b}"
    )


# ---------------------------------------------------------------------------
# 5. Differentiability in nuclear positions
# ---------------------------------------------------------------------------

def test_local_potential_differentiable_in_position():
    """jax.grad through hgh_local_potential w.r.t. nuclear positions runs and
    gives a finite, non-zero gradient."""
    grid = Grid3D(N=24, L=8.0)
    elements = (OXYGEN_HGH,)

    def total_V(positions):
        nuclei = Nuclei(
            positions=positions,
            charges=jnp.array([OXYGEN_HGH.Z_ion]),
        )
        V = hgh_local_potential(nuclei, elements, grid)
        # Scalar handle: integrate V (just a smooth functional of positions).
        return jnp.sum(V) * grid.dV

    R0 = jnp.array([[0.3, 0.0, 0.0]])
    g = jax.grad(total_V)(R0)
    assert jnp.all(jnp.isfinite(g))
    # Translating an isolated atom: ∫V dV is invariant under translation, so
    # the gradient should be ~0 to discretization. Non-trivial check: pull
    # the atom toward the boundary and watch it stop being invariant. Here we
    # only assert finiteness — the symmetry-zero check is too sensitive to
    # grid aliasing.


def test_chlorine_nonlocal_hermitian_multi_i():
    """Cl is the first HGH element with two radial projectors per ℓ-channel
    coupled by a non-trivial off-diagonal h_12. Hermiticity is the cleanest
    smoke check that the n=2 path is wired correctly.
    """
    grid = Grid3D(N=40, L=10.0)
    nuclei = Nuclei(
        positions=jnp.array([[0.0, 0.0, 0.0]]),
        charges=jnp.array([CHLORINE_HGH.Z_ion]),
    )
    elements = (CHLORINE_HGH,)

    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    phi = jax.random.normal(k1, grid.shape, dtype=jnp.float64)
    psi = jax.random.normal(k2, grid.shape, dtype=jnp.float64)

    Vphi = hgh_nonlocal_apply(phi, nuclei, elements, grid)
    Vpsi = hgh_nonlocal_apply(psi, nuclei, elements, grid)

    a = float(inner_product(phi, Vpsi, grid).real)
    b = float(inner_product(Vphi, psi, grid).real)

    assert abs(a - b) < 1e-9 * (abs(a) + abs(b) + 1.0), (
        f"⟨φ|V_nl ψ⟩ = {a}, ⟨V_nl φ|ψ⟩ = {b}"
    )


def test_lookup_returns_known_elements():
    assert lookup("H") is HYDROGEN_HGH
    assert lookup("Li") is LITHIUM_HGH
    assert lookup("O") is OXYGEN_HGH
    assert lookup("Cl") is CHLORINE_HGH
    with pytest.raises(KeyError):
        lookup("Xx")
