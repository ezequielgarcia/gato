"""End-to-end Phase 4 smoke test: H₂ closed-shell RHF + geometry optimization.

Reference (RHF/CBS, all-electron):
    R_e ≈ 1.40 a₀,   E(R_e) ≈ -1.133 E_h.

On a coarse grid with softened Coulomb (ε = h/2) the minimum sits a bit
above the CBS reference and the bond length is shifted by softening. We
accept a wide bracket on R_e and only check that the optimizer moves in
the correct direction and that the bound state lies below the
dissociation limit of two separated H atoms (-1.0 E_h).
"""
import pytest

from gato.geometry import bond_length, diatomic_along_z
from gato.grid import Grid3D
from gato.physics.h2 import (
    bo_energy,
    optimize_geometry,
    solve_rhf_at_geometry,
)


@pytest.mark.slow
def test_h2_rhf_single_point_is_bound():
    """At the experimental geometry, RHF must give a bound H₂ (E < -1.0)."""
    grid = Grid3D(N=40, L=10.0)
    nuclei = diatomic_along_z(1.4, Z1=1.0)
    point = solve_rhf_at_geometry(
        nuclei, grid, max_iters=25, mixing=0.7, order=4,
    )
    assert point.converged, f"SCF did not converge; iters={point.n_iters}"
    # Two separated H atoms: E_∞ = -1.0 (each at -0.5). H₂ at R≈R_e must lie
    # below this; exact RHF/CBS is ~-1.133.
    assert point.energy < -1.0, (
        f"H₂ RHF energy {point.energy:+.4f} not below dissociation limit -1.0"
    )
    # Sanity ceiling: nothing should drag the energy below the exact non-relativistic
    # H₂ energy (~-1.174). Catches sign / decomposition bugs.
    assert point.energy > -1.30, (
        f"H₂ RHF energy {point.energy:+.4f} suspiciously below exact -1.174"
    )


@pytest.mark.slow
def test_h2_geometry_opt_recovers_bond_length():
    """Starting from a stretched R = 1.9 a₀, Adam on the BO energy must pull
    the bond toward the equilibrium ~1.4 a₀ (softened minimum sits a bit
    higher; we accept the [1.2, 1.7] bracket)."""
    grid = Grid3D(N=32, L=10.0)
    initial = diatomic_along_z(1.9, Z1=1.0)
    result = optimize_geometry(
        initial, grid,
        geom_steps=15,
        learning_rate=0.05,
        scf_max_iters=20,
        scf_mixing=0.7,
        order=4,
        verbose=False,
    )
    R_final = float(bond_length(result.final.nuclei, 0, 1))
    assert 1.2 <= R_final <= 1.7, f"R_final = {R_final}, expected in [1.2, 1.7]"
    # The optimizer must have moved meaningfully from the start.
    assert abs(R_final - 1.9) > 0.1
    # Final geometry must still be bound below dissociation.
    assert result.final.energy < -1.0


@pytest.mark.slow
def test_h2_hellmann_feynman_matches_finite_difference():
    """The HF force from `jax.grad(bo_energy)` must agree with a symmetric
    finite difference of the BO energy at fixed orbitals."""
    import jax

    grid = Grid3D(N=28, L=10.0)
    nuclei = diatomic_along_z(1.6, Z1=1.0)
    point = solve_rhf_at_geometry(
        nuclei, grid, max_iters=20, mixing=0.7, order=4,
    )
    eps = grid.h / 2

    grad_auto = jax.grad(bo_energy, argnums=0)(
        point.nuclei.positions, point.nuclei.charges, point.orbitals,
        grid, eps, 4,
    )

    delta = 1e-3
    plus = point.nuclei.positions.at[1, 2].add(delta)
    minus = point.nuclei.positions.at[1, 2].add(-delta)
    E_plus = float(bo_energy(plus, point.nuclei.charges, point.orbitals, grid, eps, 4))
    E_minus = float(bo_energy(minus, point.nuclei.charges, point.orbitals, grid, eps, 4))
    grad_fd_z = (E_plus - E_minus) / (2 * delta)
    assert float(grad_auto[1, 2]) == pytest.approx(grad_fd_z, rel=1e-3, abs=1e-4)
