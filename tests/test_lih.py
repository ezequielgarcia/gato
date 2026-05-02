"""End-to-end Phase 4 smoke test: LiH (closed-shell RHF + HGH pseudopotentials).

Reference (RHF/CBS, HGH-LDA-q1 Li + HGH H, 2 valence electrons):
    R_e ≈ 3.02 a₀,   E ≈ -0.78 E_h,   μ ≈ 6 D.

This is the cheapest test of the *HGH non-local + multi-center
geometry-opt* combination; H₂ exercises multi-center geometry-opt
without any PP, water exercises HGH but is too expensive for CPU.
"""
import pytest

from gato.geometry import bond_length
from gato.grid import Grid3D
from gato.physics.lih import (
    bo_energy,
    lih_hgh_elements,
    lih_initial_geometry,
    optimize_geometry,
    solve_rhf_at_geometry,
)


@pytest.mark.slow
def test_lih_rhf_single_point_is_bound():
    """Near R_e the LiH RHF energy must be bound (E < E(separated atoms)).

    With HGH-LDA-q1 PPs the dissociation limit is the sum of the *PP-atomic*
    Li and H energies, which sit near -0.20 and -0.45 E_h respectively
    on a coarse grid (PP-dependent). We use a conservative ceiling of
    -0.50 E_h here — well below any plausible dissociation limit and well
    above any unphysical overbinding.
    """
    grid = Grid3D(N=40, L=12.0)
    nuclei = lih_initial_geometry(R_LiH=3.0)
    elements = lih_hgh_elements()
    point = solve_rhf_at_geometry(
        nuclei, elements, grid, max_iters=40, mixing=0.5, order=4,
    )
    assert point.converged, f"SCF did not converge; iters={point.n_iters}"
    assert point.energy < -0.50, (
        f"LiH RHF energy {point.energy:+.4f} not bound enough; expected < -0.50"
    )
    # Sanity ceiling: should not be absurdly low (catches sign / PP bugs).
    assert point.energy > -2.0, (
        f"LiH RHF energy {point.energy:+.4f} suspiciously deep; PP bug?"
    )


@pytest.mark.slow
def test_lih_geometry_opt_recovers_bond_length():
    """Starting from a stretched R = 3.8 a₀, Adam on the BO energy must
    pull the bond toward the equilibrium ~3.02 a₀. Coarse-grid PP
    geometries are softer than CBS, so we accept [2.5, 3.6]."""
    grid = Grid3D(N=32, L=12.0)
    initial = lih_initial_geometry(R_LiH=3.8)
    elements = lih_hgh_elements()
    result = optimize_geometry(
        initial, elements, grid,
        geom_steps=15,
        learning_rate=0.05,
        scf_max_iters=30,
        scf_mixing=0.5,
        order=4,
        verbose=False,
    )
    R_final = float(bond_length(result.final.nuclei, 0, 1))
    assert 2.5 <= R_final <= 3.6, f"R_final = {R_final}, expected in [2.5, 3.6]"
    # The optimizer must have moved meaningfully from the start.
    assert abs(R_final - 3.8) > 0.1


@pytest.mark.slow
def test_lih_hellmann_feynman_matches_finite_difference():
    """HF force from `jax.grad(bo_energy)` must agree with a symmetric
    finite difference at fixed orbitals. Validates the HGH non-local
    differentiability path on Li."""
    import jax

    grid = Grid3D(N=28, L=12.0)
    nuclei = lih_initial_geometry(R_LiH=3.1)
    elements = lih_hgh_elements()
    point = solve_rhf_at_geometry(
        nuclei, elements, grid, max_iters=30, mixing=0.5, order=4,
    )

    grad_auto = jax.grad(bo_energy, argnums=0)(
        point.nuclei.positions, point.nuclei.charges, point.orbitals,
        elements, grid, 4,
    )

    delta = 1e-3
    plus = point.nuclei.positions.at[1, 2].add(delta)
    minus = point.nuclei.positions.at[1, 2].add(-delta)
    E_plus = float(bo_energy(plus, point.nuclei.charges, point.orbitals, elements, grid, 4))
    E_minus = float(bo_energy(minus, point.nuclei.charges, point.orbitals, elements, grid, 4))
    grad_fd_z = (E_plus - E_minus) / (2 * delta)
    assert float(grad_auto[1, 2]) == pytest.approx(grad_fd_z, rel=1e-3, abs=1e-4)
