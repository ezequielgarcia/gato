"""Log-radial solver with nonzero angular momentum.

Verifies the centrifugal-barrier extension of the Phase 1 radial solver:
eigenvalues in the 2p, 3p, 3d channels, parity-aware inner boundary, and
the radial dipole integral driving the 1s → 2p hydrogen transition.
"""
import math

import jax.numpy as jnp
import pytest

from gato.physics.radial_hydrogen import (
    LogRadialGrid,
    radial_dipole,
    solve_bound_states,
)


@pytest.mark.parametrize(
    "ell,n_to_energy",
    [
        # Hydrogen is ℓ-degenerate: E_n = -1/(2n²), n ≥ ℓ + 1.
        (1, {2: -0.125,    3: -1.0 / 18, 4: -0.03125}),   # 2p, 3p, 4p
        (2, {3: -1.0 / 18, 4: -0.03125, 5: -0.02}),       # 3d, 4d, 5d
        (3, {4: -0.03125,  5: -0.02,    6: -1.0 / 72}),   # 4f, 5f, 6f
    ],
)
def test_hydrogen_ell_channel_energies(ell, n_to_energy):
    """E_{nℓ} = −1/(2n²) for ℓ ≥ 1 on the log-radial grid.

    r_max grows ∝ n² so higher shells don't leak into the hard wall at the
    outer boundary. N=800 with a tight inner r_min resolves the r^(ℓ+1)
    origin behavior cleanly for ℓ up to 3.
    """
    grid = LogRadialGrid(N=800, r_min=1e-3, r_max=150.0)
    # The ℓ-channel has n ≥ ℓ + 1, so the solver's eigenvalues start at n = ℓ+1.
    energies, _ = solve_bound_states(grid, Z=1.0, K=6, ell=ell)
    for n, expected in n_to_energy.items():
        idx = n - (ell + 1)  # eigenvalue index for principal quantum number n
        got = float(energies[idx])
        assert abs(got - expected) < 1e-4, (
            f"ℓ={ell}, n={n}: E = {got:.6f}, expected {expected:.6f}"
        )


def test_hydrogen_2p_is_degenerate_with_2s():
    """E_{2s} = E_{2p} for pure Coulomb — the accidental ℓ-degeneracy.

    Agreement at the 1e-4 Hartree level confirms both channels resolve
    the same n=2 eigenenergy, which is the direct observable justifying
    the centrifugal-barrier machinery.
    """
    grid = LogRadialGrid(N=800, r_min=1e-3, r_max=80.0)
    E_s, _ = solve_bound_states(grid, Z=1.0, K=2, ell=0)
    E_p, _ = solve_bound_states(grid, Z=1.0, K=1, ell=1)
    E_2s = float(E_s[1])
    E_2p = float(E_p[0])
    assert abs(E_2s - E_2p) < 1e-4, f"E_2s = {E_2s}, E_2p = {E_2p}"
    assert abs(E_2s - (-0.125)) < 1e-4
    assert abs(E_2p - (-0.125)) < 1e-4


def test_radial_dipole_1s_to_2p_matches_analytic():
    """⟨u_{2p} | r | u_{1s}⟩ = 256/(81√6) ≈ 1.2903 a₀.

    Bethe–Salpeter §63. Multiplying by the angular Gaunt piece
    ⟨Y_{10}|cosθ|Y_{00}⟩ = 1/√3 gives the famous 128√2/243 ≈ 0.7449
    total z-dipole — independent of the Cartesian-grid version tested in
    `test_spectra.py`, using only the log-radial solver.
    """
    grid = LogRadialGrid(N=800, r_min=1e-3, r_max=80.0)
    _, u_1s_set = solve_bound_states(grid, Z=1.0, K=1, ell=0)
    _, u_2p_set = solve_bound_states(grid, Z=1.0, K=1, ell=1)
    u_1s = u_1s_set[:, 0]
    u_2p = u_2p_set[:, 0]

    # Fix sign convention: pick the "positive in the interior" phase.
    # The dominant lobe of u_{1s} ∝ r e^(-r) peaks near r = 1; u_{2p} ∝
    # r² e^(-r/2) peaks near r = 4. Signing both positive there gives a
    # positive radial dipole, matching the analytic convention.
    r = grid.r()
    def _positive_at(u, r_target):
        j = int(jnp.argmin(jnp.abs(r - r_target)))
        return u if float(u[j]) > 0 else -u
    u_1s = _positive_at(u_1s, 1.0)
    u_2p = _positive_at(u_2p, 4.0)

    got = float(radial_dipole(u_2p, u_1s, grid))
    expected = 256.0 / (81.0 * math.sqrt(6.0))  # ≈ 1.29027
    assert abs(got - expected) < 1e-3, (
        f"radial dipole = {got:.5f}, expected {expected:.5f}"
    )

    # And the product with the angular Gaunt factor 1/√3 is the textbook
    # 128√2/243 — explicit cross-check against the analytic total.
    total = got / math.sqrt(3.0)
    analytic_total = 128.0 * math.sqrt(2.0) / 243.0
    assert abs(total - analytic_total) < 1e-3


def test_ell_eq_zero_still_passes_hydrogen_ground_state():
    """Regression: adding ℓ>0 support must not break the ℓ=0 path."""
    grid = LogRadialGrid(N=400, r_min=0.01, r_max=40.0)
    E, _ = solve_bound_states(grid, Z=1.0, K=1, ell=0)
    assert abs(float(E[0]) - (-0.5)) < 1e-4
