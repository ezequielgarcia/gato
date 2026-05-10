"""End-to-end Phase 4 smoke test: HCl (closed-shell RHF + HGH pseudopotentials).

Reference (RHF/CBS, HGH-LDA H + HGH-LDA-q7 Cl, 8 valence electrons):
    R_e ≈ 2.41 a₀,   E ≈ -15.5 E_h,   μ ≈ 1.5 D.

HCl is the cheapest exercise of the **multi-radial-projector HGH**
machinery: Cl's s-channel ships TWO radial projectors with a non-trivial
off-diagonal h₁₂ coupling, a path no other shipped element drives.
"""
import pytest

from gato.geometry import bond_length
from gato.grid import Grid3D
from gato.physics.hcl import (
    bo_energy,
    hcl_hgh_elements,
    hcl_initial_geometry,
    optimize_geometry,
    solve_rhf_at_geometry,
)


@pytest.mark.slow
def test_hcl_rhf_single_point_is_bound():
    """Near R_e the HCl RHF energy must be bound and physically sensible.

    With HGH-LDA-q7 Cl + HGH H the dissociation limit sits near
    E(Cl ion) + E(H) ≈ -15 E_h on a coarse grid; the bound molecule should
    sit a few mHa to ~100 mHa lower depending on grid resolution. We use
    bracket [-20, -10] E_h: tight enough to catch sign / PP bugs, loose
    enough to tolerate coarse-grid bias.
    """
    grid = Grid3D(N=24, L=12.0)
    nuclei = hcl_initial_geometry(R_HCl=2.5)
    elements = hcl_hgh_elements()
    point = solve_rhf_at_geometry(
        nuclei, elements, grid, max_iters=60, mixing=0.4, order=4,
        lanczos_iters=60,
    )
    assert -20.0 < point.energy < -10.0, (
        f"HCl RHF energy {point.energy:+.4f} out of physical range; "
        f"expected ~-15 E_h (PP-dependent on a coarse grid)."
    )
    # Four doubly-occupied valence MOs: 1 deep s-like + 3 p-like, all bound.
    assert point.orbital_energies.shape == (4,)
    assert all(float(e) < 0 for e in point.orbital_energies), (
        f"all 4 occupied orbitals must be bound; got {point.orbital_energies}"
    )


@pytest.mark.slow
def test_hcl_hellmann_feynman_matches_finite_difference():
    """HF force from `jax.grad(bo_energy)` must agree with a symmetric
    finite difference at fixed orbitals. Validates the differentiability
    of the multi-i HGH non-local block on Cl — the code path that no
    other shipped molecule exercises.
    """
    import jax

    grid = Grid3D(N=24, L=12.0)
    nuclei = hcl_initial_geometry(R_HCl=2.5)
    elements = hcl_hgh_elements()
    point = solve_rhf_at_geometry(
        nuclei, elements, grid, max_iters=40, mixing=0.4, order=4,
        lanczos_iters=60,
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


@pytest.mark.slow
def test_hcl_geometry_opt_moves_toward_equilibrium():
    """Starting from a stretched R = 3.0 a₀, a few Adam steps on the BO
    energy must pull the bond toward equilibrium. Coarse-grid PP geometries
    are softer than CBS, so we accept a generous final bracket.
    """
    grid = Grid3D(N=24, L=12.0)
    initial = hcl_initial_geometry(R_HCl=3.0)
    elements = hcl_hgh_elements()
    result = optimize_geometry(
        initial, elements, grid,
        geom_steps=10,
        learning_rate=0.05,
        scf_max_iters=40,
        scf_mixing=0.4,
        order=4,
        verbose=False,
    )
    R_final = float(bond_length(result.final.nuclei, 0, 1))
    assert 1.8 <= R_final <= 3.0, (
        f"R_final = {R_final:.4f}, expected to move toward [1.8, 3.0]"
    )
    # The optimizer must have moved meaningfully from the start.
    assert R_final < 3.0 - 0.05, f"bond did not contract: R_final = {R_final}"
