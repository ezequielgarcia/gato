"""Phase 6 — atomic absorption spectra.

First slice: verify the spectra primitives (transition dipole, oscillator
strength) against analytic hydrogen values, and reproduce the Balmer series
line positions from the log-radial eigenvalues.
"""
import math

import jax.numpy as jnp
import pytest

from gato.grid import Grid3D
from gato.physics.radial_hydrogen import LogRadialGrid, solve_bound_states
from gato.spectra import (
    HC_HARTREE_NM,
    einstein_A,
    oscillator_strength,
    photon_wavelength_nm,
    transition_dipole,
)


# --- analytic hydrogen wavefunctions (atomic units, Z=1) -----------------

def _psi_1s(X, Y, Z):
    r = jnp.sqrt(X * X + Y * Y + Z * Z)
    return jnp.exp(-r) / jnp.sqrt(jnp.pi)


def _psi_2p_z(X, Y, Z):
    """2p_z, real form. ψ = (1/(4√(2π))) z e^(-r/2)."""
    r = jnp.sqrt(X * X + Y * Y + Z * Z)
    return Z * jnp.exp(-r / 2.0) / (4.0 * jnp.sqrt(2.0 * jnp.pi))


# --- tests ---------------------------------------------------------------

def test_transition_dipole_1s_2pz_hydrogen():
    """⟨1s | z | 2p_z⟩ = 128√2 / 243 ≈ 0.74490 a.u. (Bethe & Salpeter §63).

    The grid needs to be generous: the 2p extends to ~10 a₀, and the
    integrand r · ψ_1s ψ_2p peaks near r ≈ 2 a₀. L=20, N=80 gives
    better than 1% against the analytic value; tighter grids converge
    toward it monotonically.
    """
    g = Grid3D(N=80, L=20.0)
    X, Y, Z = g.coords()
    psi_1s = _psi_1s(X, Y, Z)
    psi_2pz = _psi_2p_z(X, Y, Z)

    mu = transition_dipole(psi_1s, psi_2pz, g)
    analytic_z = 128.0 * math.sqrt(2.0) / 243.0  # ≈ 0.74490

    assert abs(float(mu[0])) < 1e-10, "μ_x should vanish by parity"
    assert abs(float(mu[1])) < 1e-10, "μ_y should vanish by parity"
    assert abs(float(mu[2]) - analytic_z) < 1e-2, (
        f"μ_z = {float(mu[2]):.5f}, analytic = {analytic_z:.5f}"
    )


def test_oscillator_strength_1s_to_2p_manifold():
    """f(1s → 2p) summed over m ∈ {-1, 0, 1} equals 0.41620 exactly.

    Classic textbook value. With the analytic dipole μ_z = 128√2/243 and
    the three degenerate m-sublevels giving identical |μ|² contributions
    (symmetry; μ_x for 2p_x and μ_y for 2p_y are equal to μ_z for 2p_z),
    the sum f = 3 · (2/3) ω |μ_z|² = 2 ω |μ_z|² with ω = 3/8 Ha.
    """
    mu_z = 128.0 * math.sqrt(2.0) / 243.0
    omega = 3.0 / 8.0  # E_2p - E_1s = -1/8 - (-1/2) = 3/8 Hartree

    # single-sublevel f
    mu_vec = jnp.array([0.0, 0.0, mu_z])
    f_single = float(oscillator_strength(omega, mu_vec))
    # total 2p (sum over three degenerate sublevels)
    f_total = 3.0 * f_single

    expected_total = 2.0 * omega * mu_z * mu_z  # closed form, ≈ 0.41620
    assert abs(f_total - expected_total) < 1e-10
    # and the famous number itself
    assert abs(f_total - 0.41620) < 1e-4, f"f(1s→2p) = {f_total}"


def test_balmer_series_line_positions():
    """Balmer lines from ℓ=0 eigenvalues match the theoretical Rydberg formula.

    Hydrogen energies E_n = -1/(2n²) give λ_{n→2} = hc / (1/8 - 1/(2n²)).
    The log-radial ℓ=0 solver recovers E_n for n up to ~6 on a well-sized
    grid (needs r_max ≳ 3n² = 75 for n=5). We require agreement with the
    theoretical infinite-nuclear-mass values to within 0.1 nm.

    NIST tabulates the Balmer lines ~0.2 nm longer than this (reduced-mass
    correction m_p/(m_p+m_e) ≈ 1 − 1/1836); we are not modeling reduced
    mass so we test against the bare Schrödinger prediction.
    """
    grid = LogRadialGrid(N=800, r_min=1e-3, r_max=120.0)
    energies, _ = solve_bound_states(grid, Z=1.0, K=6)
    E = [float(e) for e in energies]

    # Sanity: E_n ≈ -1/(2n²) for n = 1..5
    for n in range(1, 6):
        expected = -1.0 / (2.0 * n * n)
        assert abs(E[n - 1] - expected) < 1e-4, (
            f"n={n}: E = {E[n - 1]:.6f}, expected {expected:.6f}"
        )

    # Balmer lines: n = 3, 4, 5 → 2.
    E2 = -1.0 / 8.0
    cases = [
        (3, 656.113),  # H-α  (theoretical, λ = hc / (5/72))
        (4, 486.009),  # H-β  (theoretical, λ = hc / (3/32))
        (5, 433.937),  # H-γ  (theoretical, λ = hc / (21/200))
    ]
    for n_upper, lam_theory in cases:
        dE = E[n_upper - 1] - E[1]  # E_n − E_2
        # also check vs analytic value computed from HC_HARTREE_NM
        analytic_dE = -1.0 / (2.0 * n_upper * n_upper) - E2
        assert abs(HC_HARTREE_NM / analytic_dE - lam_theory) < 1e-2
        lam_computed = float(photon_wavelength_nm(dE))
        assert abs(lam_computed - lam_theory) < 0.1, (
            f"n={n_upper}→2: λ = {lam_computed:.3f} nm, want {lam_theory:.3f} nm"
        )


def test_einstein_A_positive_and_scales():
    """Einstein A ∝ ω³|μ|². Check scaling and positivity."""
    mu = jnp.array([0.0, 0.0, 0.5])
    A1 = float(einstein_A(0.1, mu))
    A2 = float(einstein_A(0.2, mu))
    assert A1 > 0
    # doubling ω should multiply A by 8
    assert abs(A2 / A1 - 8.0) < 1e-10


@pytest.mark.parametrize("omega_ha,expected_nm", [
    (3.0 / 8.0, 121.502),   # Lyman-α (1s→2p), theoretical ≈ 121.5 nm
    (5.0 / 72.0, 656.113),  # H-α
])
def test_photon_wavelength_nm(omega_ha, expected_nm):
    got = float(photon_wavelength_nm(omega_ha))
    assert abs(got - expected_nm) < 5e-3
