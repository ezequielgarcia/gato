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
    solve_zora_ground_state,
)
from gato.physics.radial_hydrogen import (
    LogRadialGrid,
    solve_bound_states,
    solve_ground_state,
)
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


# -------------------- ZORA --------------------

def test_zora_hydrogen_1s_shift_is_alpha2_Z4_over_4():
    """E_ZORA − E_NR = −α² Z⁴ / 4 for hydrogen 1s at leading order in α².

    This is *twice* the Sommerfeld / Dirac shift (−α² Z⁴ / 8), which is a
    known feature of *scalar* ZORA for deeply-bound hydrogenic 1s states:
    ZORA is a different O(α²) approximation than the Foldy–Wouthuysen
    expansion that produces MV + Darwin, and it lacks the "picture change"
    correction that would halve the shift and bring it to the Dirac value.
    Both ZORA and MV+Darwin are valid *leading-order* scalar-relativistic
    corrections; they disagree on the coefficient by a factor of 2 for H
    1s and both scale as Z⁴, which is exactly what this test pins down.

    Derivation: expanding T_ZORA = p·K·p with K ≈ 1/2 + V/(4c²) gives the
    perturbative shift (α²/4)⟨p V p⟩_{1s} = (α²/4)(−Z⁴) = −α² Z⁴ / 4.
    """
    grid = LogRadialGrid(N=1600, r_min=0.01, r_max=40.0)
    E_nr, _ = solve_ground_state(grid, Z=1.0)
    E_zora, _ = solve_zora_ground_state(grid, Z=1.0)
    shift = float(E_zora - E_nr)
    expected = -ALPHA2 / 4.0  # ≈ -1.3313e-5
    assert abs(shift - expected) / abs(expected) < 1e-2, (
        f"ZORA shift = {shift:.4e}, expected {expected:.4e} (−α²/4)"
    )
    assert shift < 0


def test_zora_shift_converges_with_N():
    """ZORA shift settles to a stable value as the grid refines.

    A discretization bug (e.g. the missing (K'/r) u term) diverges
    monotonically with N; a correctly-discretized operator converges
    towards a fixed limit. We check monotone convergence over
    N = 800, 1600, 3200 by requiring the N=1600→3200 step to be smaller
    than the N=800→1600 step.
    """
    shifts = []
    for N in (800, 1600, 3200):
        grid = LogRadialGrid(N=N, r_min=0.01, r_max=40.0)
        E_nr, _ = solve_ground_state(grid, Z=1.0)
        E_zora, _ = solve_zora_ground_state(grid, Z=1.0)
        shifts.append(float(E_zora - E_nr))
    step12 = abs(shifts[1] - shifts[0])
    step23 = abs(shifts[2] - shifts[1])
    assert step23 < step12, (
        f"ZORA not converging with N: shifts = {shifts}"
    )
    # And the last value sits inside the physically-expected range.
    assert -3.0 * ALPHA2 / 8.0 < shifts[-1] < -1.0 * ALPHA2 / 8.0


def test_zora_is_roughly_twice_mv_plus_darwin():
    """Scalar ZORA ≈ 2 · (MV + Darwin) for H 1s.

    Both are O(α²) scalar-relativistic corrections to the NR Schrödinger
    energy, but ZORA and Foldy–Wouthuysen (which generates MV + Darwin)
    are distinct resummations and give coefficients that differ by a
    factor of 2 on deeply-bound 1s states. Verifying this relation
    connects the two code paths and validates both discretizations:
    a bug in either one would break the ratio, and the factor of 2 is
    sharp enough to catch subtle sign/derivative errors.
    """
    grid = LogRadialGrid(N=1600, r_min=0.01, r_max=40.0)
    E_nr, u_nr = solve_ground_state(grid, Z=1.0)
    E_zora, _ = solve_zora_ground_state(grid, Z=1.0)
    shift_zora = float(E_zora - E_nr)
    shift_pert = float(fine_structure_shift(u_nr, grid, Z=1.0))
    ratio = shift_zora / shift_pert
    assert abs(ratio - 2.0) < 0.1, (
        f"ZORA / (MV+Darwin) = {ratio:.3f}, expected ≈ 2"
    )


def test_zora_z_scaling():
    """E_ZORA − E_NR scales as Z⁴ (leading) for hydrogenic ions.

    The ratio of shifts at Z = 2 vs Z = 1 should equal 16 to leading
    order; the O(α⁴) correction enters as (Zα)⁴ so the ratio holds
    tightly for Z ≤ 10. Tolerance: 1 %.
    """
    grid1 = LogRadialGrid(N=1600, r_min=0.01, r_max=40.0)
    grid2 = LogRadialGrid(N=1600, r_min=0.005, r_max=20.0)

    E_nr_1, _ = solve_ground_state(grid1, Z=1.0)
    E_zora_1, _ = solve_zora_ground_state(grid1, Z=1.0)
    shift_1 = float(E_zora_1 - E_nr_1)

    E_nr_2, _ = solve_ground_state(grid2, Z=2.0)
    E_zora_2, _ = solve_zora_ground_state(grid2, Z=2.0)
    shift_2 = float(E_zora_2 - E_nr_2)

    ratio = shift_2 / shift_1
    assert abs(ratio - 16.0) / 16.0 < 1e-2, (
        f"Z⁴ scaling: shift(Z=2) / shift(Z=1) = {ratio:.3f}, want 16"
    )
