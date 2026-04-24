"""Phase 6 — atomic absorption spectra from bound-state wavefunctions.

The Phase 1–5 solvers return bound-state eigenpairs; Phase 6 uses the
*relations between* those states: electric-dipole transition amplitudes,
oscillator strengths, Einstein A coefficients, and photon wavelengths.

Everything here is a pure function of inputs a solver already produces.
No new SCF machinery, no new finite-difference operators — just grid
integrations and closed-form kinematic factors. Atomic units throughout
(ℏ = m_e = e = 1; c = 1/α ≈ 137.036).

References
----------
- Bethe & Salpeter, *Quantum Mechanics of One- and Two-Electron Atoms*, §61–63.
- Cowan, *The Theory of Atomic Structure and Spectra*, ch. 14.
- NIST Atomic Spectra Database: https://physics.nist.gov/asd
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .grid import Grid3D, inner_product

# Fundamental constants in atomic units.
FINE_STRUCTURE = 7.2973525693e-3  # α (dimensionless)
SPEED_OF_LIGHT_AU = 1.0 / FINE_STRUCTURE  # c in atomic units

# Unit conversions for reporting.
HARTREE_EV = 27.211386245988
BOHR_NM = 0.0529177210903
# hc in Hartree·nm: photon of energy E [Hartree] has λ [nm] = HC_HARTREE_NM / E.
#   hc = 1239.84198 eV·nm, and 1 Hartree = 27.211386 eV.
HC_HARTREE_NM = 1239.841984 / HARTREE_EV  # ≈ 45.56335


def transition_dipole(
    psi_a: jax.Array, psi_b: jax.Array, grid: Grid3D
) -> jax.Array:
    """Electric-dipole matrix element μ_{a←b} = ⟨a | r | b⟩ on a Cartesian grid.

    Returns the 3-vector (μ_x, μ_y, μ_z) in atomic units (e·a₀ = 1).
    Real for real ψ; complex in general.

    The transition-dipole *moment* driving an E1 photon absorption from
    state b (initial) to state a (final) is this integral; the oscillator
    strength and Einstein coefficients are built from |μ|² = μ·μ*.
    """
    X, Y, Z = grid.coords()
    integrand = jnp.conj(psi_a) * psi_b
    mu_x = jnp.sum(integrand * X) * grid.dV
    mu_y = jnp.sum(integrand * Y) * grid.dV
    mu_z = jnp.sum(integrand * Z) * grid.dV
    return jnp.stack([mu_x, mu_y, mu_z])


def oscillator_strength(omega: float | jax.Array, mu: jax.Array) -> jax.Array:
    """Dimensionless oscillator strength f_{0→e} = (2/3) ω |μ|².

    Parameters
    ----------
    omega : transition frequency ω = E_e − E_0 in atomic units (Hartree).
    mu : complex 3-vector transition dipole in atomic units (shape (3,)).

    In atomic units m_e = ℏ = 1 so the usual (2/3)(m_e/ℏ²)ω|μ|² collapses to
    (2/3) ω |μ|². The Thomas–Reiche–Kuhn sum rule Σ_e f_{0→e} = N_e (number
    of electrons) is the standard normalization cross-check.
    """
    mu_sq = jnp.sum(jnp.abs(mu) ** 2)
    return (2.0 / 3.0) * omega * mu_sq


def einstein_A(omega: float | jax.Array, mu: jax.Array) -> jax.Array:
    """Spontaneous-emission rate A_{e→0} = (4 ω³)/(3 c³) |μ|² (atomic units).

    In SI units this is A = (ω³ e²)/(3 π ε₀ ℏ c³) |r_{e0}|². In atomic
    units (4π ε₀ = ℏ = e = m_e = 1) it collapses to the form above, where
    c = 1/α ≈ 137.036. The natural linewidth is Γ_nat = ℏ A = A in a.u.
    """
    mu_sq = jnp.sum(jnp.abs(mu) ** 2)
    c = SPEED_OF_LIGHT_AU
    return (4.0 / 3.0) * (omega ** 3) / (c ** 3) * mu_sq


def photon_wavelength_nm(omega: float | jax.Array) -> jax.Array:
    """λ [nm] for a photon of energy ω [Hartree]. Convenience for reporting."""
    return HC_HARTREE_NM / omega
