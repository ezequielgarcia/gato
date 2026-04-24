"""Phase 7 — perturbative scalar-relativistic corrections (hydrogenic ℓ=0).

Validates the mass–velocity and Darwin expectation values on the log-radial
ground state against the Sommerfeld analytic formulas. Hydrogen 1s is the
clean reference: ⟨p⁴⟩ = 5, |ψ(0)|² = 1/π, and the total fine-structure
shift is −α²/8. For hydrogenic ions E_fs scales as Z⁴.
"""
import math

import pytest

from gato.physics.fine_structure import (
    darwin_expectation,
    fine_structure_shift,
    mass_velocity_expectation,
    p4_expectation,
    psi_at_origin_sq,
)
from gato.physics.radial_hydrogen import LogRadialGrid, solve_bound_states
from gato.spectra import FINE_STRUCTURE

ALPHA2 = FINE_STRUCTURE ** 2


def _hydrogen_1s(Z, N=1600):
    """Solve ℓ=0 ground state on a grid sized for the 1s scale of charge Z.

    r_min = 0.01/Z is the sweet spot: smaller values crowd grid points
    into r < 10⁻⁵ a₀ where the chain-rule factor 1/r'² in the log-grid
    Laplacian amplifies any 4th-order stencil noise enough to bias
    ⟨p⁴⟩ by several percent. r_min = 0.01 truncates the
    (u''(r))² dr integrand at r ≈ 10⁻⁵ — where the integrand itself is
    still near its r=0 value of 16 but the missing contribution from
    (0, r_min) is bounded by ~16 · r_min = 0.16, vastly smaller than
    the full ⟨p⁴⟩ = 5.
    """
    grid = LogRadialGrid(N=N, r_min=0.01 / Z, r_max=40.0 / Z)
    E, states = solve_bound_states(grid, Z=Z, K=1, ell=0)
    return float(E[0]), states[:, 0], grid


def test_p4_expectation_hydrogen_1s():
    """⟨p⁴⟩_{1s} = 5 Z⁴ (closed form for hydrogen ground state)."""
    for Z in (1.0, 2.0, 3.0):
        _, u, grid = _hydrogen_1s(Z)
        got = float(p4_expectation(u, grid))
        expected = 5.0 * Z ** 4
        assert abs(got - expected) / expected < 3e-3, (
            f"Z={Z}: ⟨p⁴⟩ = {got:.4f}, expected {expected:.4f}"
        )


def test_psi_at_origin_squared_hydrogen_1s():
    """|ψ(0)|²_{1s} = Z³ / π for hydrogen."""
    for Z in (1.0, 2.0, 3.0):
        _, u, grid = _hydrogen_1s(Z)
        got = float(psi_at_origin_sq(u, grid))
        expected = Z ** 3 / math.pi
        assert abs(got - expected) / expected < 3e-3, (
            f"Z={Z}: |ψ(0)|² = {got:.5f}, expected {expected:.5f}"
        )


def test_mass_velocity_matches_closed_form():
    """⟨H_MV⟩_{1s} = −5 α² Z⁴ / 8."""
    for Z in (1.0, 2.0):
        _, u, grid = _hydrogen_1s(Z)
        got = float(mass_velocity_expectation(u, grid))
        expected = -5.0 * ALPHA2 * Z ** 4 / 8.0
        assert abs(got - expected) / abs(expected) < 3e-3, (
            f"Z={Z}: ⟨H_MV⟩ = {got:.4e}, expected {expected:.4e}"
        )


def test_darwin_matches_closed_form():
    """⟨H_D⟩_{1s} = α² Z⁴ / 2."""
    for Z in (1.0, 2.0):
        _, u, grid = _hydrogen_1s(Z)
        got = float(darwin_expectation(u, grid, Z=Z))
        expected = 0.5 * ALPHA2 * Z ** 4
        assert abs(got - expected) / expected < 3e-3, (
            f"Z={Z}: ⟨H_D⟩ = {got:.4e}, expected {expected:.4e}"
        )


def test_total_fine_structure_shift_sommerfeld():
    """Total scalar-relativistic shift for H 1s: −α² / 8 ≈ −6.6566 × 10⁻⁶ Ha.

    This is the Sommerfeld fine-structure energy at (n=1, j=1/2). Spin–orbit
    vanishes for ℓ=0 so the MV + Darwin sum *is* the full leading-α² shift
    (no spin piece missing). Z⁴ scaling verified by running Z = 1, 2.

    Tolerance: MV and Darwin each converge at ~0.2% on the N=1600 grid;
    their sum is dominated by the partial cancellation (MV/|shift| ≈ 5,
    D/|shift| ≈ 4), so the same 0.2% absolute error in each becomes ~2%
    relative error on the combined shift. This is fundamental to the
    subtraction, not a solver defect.
    """
    for Z in (1.0, 2.0):
        E_nr, u, grid = _hydrogen_1s(Z)
        shift = float(fine_structure_shift(u, grid, Z=Z))
        expected = -ALPHA2 * Z ** 4 / 8.0
        assert abs(shift - expected) / abs(expected) < 3e-2, (
            f"Z={Z}: ΔE_fs = {shift:.4e}, expected {expected:.4e}"
        )
        # And the shift must be negative (attractive relativistic binding).
        assert shift < 0


def test_fine_structure_is_small_compared_to_binding():
    """|ΔE_fs| ≪ |E_nr|: the perturbation is self-consistently small (α² ≈ 1/19000)."""
    E_nr, u, grid = _hydrogen_1s(Z=1.0)
    shift = float(fine_structure_shift(u, grid, Z=1.0))
    # Expect |shift/E_nr| ≈ α²/4 ≈ 1.33e-5
    ratio = abs(shift / E_nr)
    assert ratio < 1e-4
    assert ratio > 1e-6
