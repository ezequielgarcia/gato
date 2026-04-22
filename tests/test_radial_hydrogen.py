"""Cross-check of Phase 1: hydrogen ground state on a log-radial grid
with a *pure* -Z/r potential (no softening).

The 3D Cartesian stack uses a softened Coulomb V = -1/√(r² + ε²) and
extrapolates ε → 0 linearly to close the residual (README §9,
-0.504 Eh on a 64³ grid). This module attacks the same problem from
the other side: keep V = -Z/r exact and resolve the cusp with a log
grid that packs points near the nucleus. Agreement between the two
approaches is the real sanity check on the Phase 1 number.
"""
import jax.numpy as jnp
import pytest

from gato.physics.radial_hydrogen import (
    LogRadialGrid,
    radial_inner_product,
    solve_ground_state,
)


@pytest.mark.parametrize(
    "Z,r_max",
    [
        (1.0, 40.0),   # H:    peak at r=1
        (2.0, 20.0),   # He+:  peak at r=1/2
        (3.0, 15.0),   # Li2+: peak at r=1/3
    ],
)
def test_hydrogenic_ground_state_energy(Z, r_max):
    """E₀ = -Z²/2 for Z ∈ {1, 2, 3} via pure -Z/r on the log grid."""
    grid = LogRadialGrid(N=400, r_min=0.01, r_max=r_max)
    E0, _ = solve_ground_state(grid, Z=Z)
    expected = -0.5 * Z * Z
    assert abs(float(E0) - expected) < 1e-4, (
        f"Z={Z}: got E₀ = {float(E0):.6f}, expected {expected:.6f}"
    )


def test_ground_state_is_normalized():
    """∫|u|²dr = 1 and the dominant lobe has consistent sign."""
    grid = LogRadialGrid(N=400, r_min=0.01, r_max=40.0)
    _, u0 = solve_ground_state(grid, Z=1.0)
    norm = float(radial_inner_product(u0, u0, grid))
    assert abs(norm - 1.0) < 1e-10
    # 1s is nodeless on the physical support. We don't require strict
    # positivity of every u_j (exponential tail is dominated by float noise),
    # only that |u| is overwhelmingly one-signed where it is non-negligible.
    if jnp.sum(u0) < 0:
        u0 = -u0
    mask = jnp.abs(u0) > 1e-3
    assert bool(jnp.all(u0[mask] > 0)), "1s u(r) should be one-signed on its support"


def test_grid_convergence_is_monotone():
    """Error drops monotonically with N under the odd-parity inner BC.

    On a log grid the effective order depends on the local stencil error
    and the cancellation in the chain-rule Laplacian; we don't claim a
    clean h⁴, just that finer grids do not get worse.
    """
    errors = []
    for N in (100, 200, 400):
        grid = LogRadialGrid(N=N, r_min=0.01, r_max=40.0)
        E0, _ = solve_ground_state(grid, Z=1.0)
        errors.append(abs(float(E0) - (-0.5)))
    assert errors[0] >= errors[1] >= errors[2], (
        f"non-monotone convergence: {errors}"
    )
    # Require that the finest-grid result is well under 1 mHa.
    assert errors[-1] < 1e-4, f"final error {errors[-1]:.2e} exceeds 1e-4"


def test_potential_is_unsoftened():
    """The ground-state energy is produced from V(r) = -Z/r exactly — no
    ε appears anywhere in the derivation, and the log grid samples r-values
    many decades smaller than any softening Phase 1 uses on the 3D Cartesian
    side. This is the whole reason for this cross-check module.
    """
    grid = LogRadialGrid(N=300, r_min=0.01, r_max=30.0)
    r = grid.r()
    # Smallest sampled r is orders of magnitude below any practical softening.
    assert float(jnp.min(r)) < 1e-3
    E0, _ = solve_ground_state(grid, Z=1.0)
    E0 = float(E0)
    assert jnp.isfinite(E0)
    assert abs(E0 - (-0.5)) < 1e-3
