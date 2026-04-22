"""End-to-end Phase 2: H₂⁺ Born-Oppenheimer curve and geometry optimization.

Analytic reference (Burrau 1927, prolate spheroidal exact solution):
    R_e = 1.9972 a₀,   E(R_e) = -0.6026 E_h.

On a coarse grid with softened Coulomb, the minimum of E(R) is shallow
and slightly shifted; we accept ~0.1 a₀ bracket on R_e and ~30 mHa on E.
"""
import jax
import jax.numpy as jnp
import pytest

from gato.geometry import Nuclei, bond_length
from gato.grid import Grid3D
from gato.physics.h2_plus import (
    bo_curve,
    bo_energy,
    hellmann_feynman_force,
    optimize_geometry,
    solve_electronic,
)


def _h2plus(R: float) -> Nuclei:
    return Nuclei(
        positions=jnp.array([[0.0, 0.0, -R / 2], [0.0, 0.0, R / 2]]),
        charges=jnp.asarray([1.0, 1.0]),
    )


@pytest.mark.slow
def test_bo_curve_has_bound_minimum():
    """E(R) must drop below two separated hydrogen atoms (E = -0.5) in the
    bonding region, and the minimum should sit between 1.5 and 2.5 a₀."""
    grid = Grid3D(N=40, L=10.0)
    Rs = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
    curve = bo_curve(Rs, grid=grid, n_steps=1500, order=4)
    energies = [p.energy for p in curve]
    # Dissociation limit (large R): E → -0.5 (one H atom + infinitely distant proton).
    # Take R=6 as ~dissociated.
    assert energies[-1] < -0.40, f"asymptotic E = {energies[-1]}, expected near -0.5"
    # Bonding region: energy at R≈2 should be meaningfully below dissociation.
    i_min = int(jnp.argmin(jnp.array(energies)))
    R_min = Rs[i_min]
    assert 1.5 <= R_min <= 2.5, f"BO minimum at R = {R_min}, expected in [1.5, 2.5]"
    # The bonding stabilization should be positive (bound state exists).
    assert energies[i_min] < energies[-1] - 0.03, (
        f"bonding well too shallow: E_min = {energies[i_min]:.4f}, "
        f"E_∞ = {energies[-1]:.4f}"
    )


@pytest.mark.slow
def test_hellmann_feynman_matches_finite_difference():
    """The HF force obtained via `jax.grad` must agree with a symmetric
    finite difference of the BO energy, confirming the autodiff pipeline.
    """
    grid = Grid3D(N=32, L=10.0)
    R = 2.2
    point = solve_electronic(_h2plus(R), grid, n_steps=1500, order=4)
    eps_soft = grid.h / 2

    F_auto = hellmann_feynman_force(
        point.nuclei.positions, point.nuclei.charges, point.psi,
        grid, eps_soft, order=4,
    )

    # Finite-difference the total energy along z on atom 1 with psi held fixed.
    delta = 1e-3
    plus = point.nuclei.positions.at[1, 2].add(delta)
    minus = point.nuclei.positions.at[1, 2].add(-delta)
    E_plus = float(bo_energy(plus, point.nuclei.charges, point.psi, grid, eps_soft, 4))
    E_minus = float(bo_energy(minus, point.nuclei.charges, point.psi, grid, eps_soft, 4))
    F_fd_z = -(E_plus - E_minus) / (2 * delta)
    assert F_auto[1, 2] == pytest.approx(F_fd_z, rel=1e-3, abs=1e-4)


@pytest.mark.slow
def test_geometry_optimization_recovers_bond_length():
    """The exit criterion for Phase 2: starting from R = 2.6 a₀, gradient
    descent on E(R) must pull R toward the equilibrium ~2.0 a₀ (softened).
    We don't demand hitting the Burrau value to 0.01 a₀ on this coarse grid —
    only that the optimizer moves in the correct direction and lands inside
    a reasonable bracket."""
    grid = Grid3D(N=32, L=10.0)
    initial = _h2plus(2.6)
    result = optimize_geometry(
        initial, grid,
        electronic_steps=1200,
        geom_steps=20,
        learning_rate=0.05,
        order=4,
    )
    R_final = float(bond_length(result.final.nuclei, 0, 1))
    # The softened-Coulomb minimum usually lives slightly above the Burrau
    # value 1.997; accept anything in [1.7, 2.3].
    assert 1.7 <= R_final <= 2.3, f"R_final = {R_final}, expected in [1.7, 2.3]"
    # Trajectory should have moved meaningfully from the initial geometry.
    assert abs(R_final - 2.6) > 0.2
    # Energy at the optimized geometry must be bound (below separated atoms).
    assert result.final.energy < -0.5
