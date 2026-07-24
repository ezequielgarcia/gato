"""End-to-end Phase 4 smoke test: H₂O (closed-shell RHF + HGH pseudopotentials).

Reference (RHF/CBS, HGH-LDA O q6 + HGH-LDA H, 8 valence electrons):
    R_OH ≈ 1.81 a₀ (0.957 Å),   ∠HOH ≈ 104.5°,   E ≈ -17 E_h (PP-dependent).

Water is the headline GATO demo: three nuclei, four doubly-occupied MOs,
and the O block exercises the single-radial-projector HGH s-channel. This
suite is the water counterpart to `test_hcl.py`; both are `@slow` because
the end-to-end SCF on a 24³ grid is wall-clock heavy for per-push CI.
"""
import pytest

from gato.geometry import bond_angle, bond_length
from gato.grid import Grid3D
from gato.physics.water import (
    bo_energy,
    optimize_water_geometry,
    solve_rhf_at_geometry,
    water_hgh_elements,
    water_initial_geometry,
)


@pytest.mark.slow
def test_water_rhf_single_point_is_bound():
    """Near equilibrium the H₂O RHF energy must be bound and physical.

    The absolute HGH valence energy sits near -17 to -18.5 E_h depending on
    grid resolution (the narrow O s-projector, r_ℓ ≈ 0.22 a₀, is only
    marginally resolved on a coarse grid, which biases the total). Bracket
    [-25, -12] is tight enough to catch a sign / PP-alignment bug, loose
    enough to tolerate coarse-grid bias.
    """
    grid = Grid3D(N=24, L=12.0)
    nuclei = water_initial_geometry(R_OH=1.85, angle_deg=104.0)
    elements = water_hgh_elements()
    point = solve_rhf_at_geometry(
        nuclei, elements, grid, max_iters=80, mixing=0.4, order=4,
        lanczos_iters=60,
    )
    assert -25.0 < point.energy < -12.0, (
        f"H₂O RHF energy {point.energy:+.4f} out of physical range; "
        f"expected ~-17 E_h (PP-dependent on a coarse grid)."
    )
    # Four doubly-occupied valence MOs (2a₁, 1b₂, 3a₁, 1b₁), all bound.
    assert point.orbital_energies.shape == (4,)
    assert all(float(e) < 0 for e in point.orbital_energies), (
        f"all 4 occupied orbitals must be bound; got {point.orbital_energies}"
    )


@pytest.mark.slow
def test_water_hellmann_feynman_matches_finite_difference():
    """HF force from `jax.grad(bo_energy)` must agree with a symmetric
    finite difference at fixed orbitals. Validates differentiability of the
    HGH local + non-local blocks through the three-center water geometry;
    we probe the z-coordinate of one hydrogen.
    """
    import jax

    grid = Grid3D(N=24, L=12.0)
    nuclei = water_initial_geometry(R_OH=1.85, angle_deg=104.0)
    elements = water_hgh_elements()
    point = solve_rhf_at_geometry(
        nuclei, elements, grid, max_iters=60, mixing=0.4, order=4,
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
def test_water_geometry_opt_lowers_energy():
    """A few Adam steps on the BO energy from a perturbed start must go
    *downhill* (not raise the total energy) and keep the geometry physical.

    We deliberately do NOT assert a bond-length direction. On a coarse 24³
    grid (h ≈ 0.5 a₀, far coarser than the O s-projector width r_ℓ ≈ 0.22)
    the PP equilibrium is grid-shifted to a longer bond than the 1.81 a₀
    CBS value, so "moves toward 1.81" is the wrong invariant here. The
    meaningful, grid-independent check is that optimization decreases the
    energy — which, with the Hellmann–Feynman force validated separately,
    certifies the geometry loop is wired correctly.
    """
    import math

    grid = Grid3D(N=24, L=12.0)
    initial = water_initial_geometry(R_OH=1.60, angle_deg=96.0)
    elements = water_hgh_elements()
    E0 = solve_rhf_at_geometry(
        initial, elements, grid, max_iters=40, mixing=0.4, order=4,
        lanczos_iters=60,
    ).energy
    result = optimize_water_geometry(
        initial, elements, grid,
        geom_steps=10,
        learning_rate=0.05,
        scf_max_iters=40,
        scf_mixing=0.4,
        order=4,
        verbose=False,
    )
    R_final = float(bond_length(result.final.nuclei, 0, 1))
    angle_deg = math.degrees(float(bond_angle(result.final.nuclei, 1, 0, 2)))

    # Optimization must not increase the energy (tiny slack for Adam wobble).
    assert result.final.energy <= E0 + 1e-3, (
        f"geometry opt raised energy: E0 = {E0:+.6f} -> "
        f"E_final = {result.final.energy:+.6f}"
    )
    # Geometry must stay physical (no collapse / inversion / runaway).
    assert 1.2 <= R_final <= 3.0, f"R_OH = {R_final:.4f} a₀ unphysical"
    assert 80.0 <= angle_deg <= 130.0, f"∠HOH = {angle_deg:.2f}° unphysical"
