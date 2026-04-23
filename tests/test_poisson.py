"""Tests for the FFT-based Hartree solver.

Key analytic check: for a normalized isotropic Gaussian density
    ρ(r) = (α/π)^(3/2) exp(-α r²),     ∫ρ dV = 1,
the Hartree potential is known in closed form,
    J(r) = erf(√α · r) / r,            J(0) = 2 √(α/π),
and the Hartree self-energy is
    E_H = (1/2) ∫ρ J dV = (1/2) √(2α/π).

Softening the 1/r kernel by ε ≲ h/2 perturbs the potential only at short range,
so large-r values and the total energy match to ~1% on well-resolved grids.
"""
import jax.numpy as jnp
import numpy as np

from gato.grid import Grid3D, integrate
from gato.solvers import hartree_energy, hartree_potential


def _gaussian(grid: Grid3D, alpha: float, center=(0.0, 0.0, 0.0)) -> jnp.ndarray:
    X, Y, Z = grid.coords()
    x0, y0, z0 = center
    r2 = (X - x0) ** 2 + (Y - y0) ** 2 + (Z - z0) ** 2
    return (alpha / jnp.pi) ** 1.5 * jnp.exp(-alpha * r2)


def test_gaussian_hartree_energy_matches_analytic():
    grid = Grid3D(N=48, L=10.0)
    alpha = 0.5
    rho = _gaussian(grid, alpha)

    # sanity: density integrates to 1
    assert abs(float(integrate(rho, grid)) - 1.0) < 1e-4

    E = float(hartree_energy(rho, grid))
    E_exact = 0.5 * float(jnp.sqrt(2.0 * alpha / jnp.pi))
    # softening ε=h/2 gives ~1% systematic error on the self-energy
    assert abs(E - E_exact) / E_exact < 1e-2


def test_gaussian_hartree_potential_far_field():
    """Far from the charge, J(r) → erf(√α r)/r regardless of softening."""
    grid = Grid3D(N=80, L=10.0)
    alpha = 0.5
    rho = _gaussian(grid, alpha)
    from scipy.special import erf

    J = np.asarray(hartree_potential(rho, grid))
    X, Y, Z = (np.asarray(a) for a in grid.coords())
    r = np.sqrt(X**2 + Y**2 + Z**2)

    # sample a shell well outside the Gaussian width (σ = 1/√(2α) = 1)
    mask = (r > 2.5) & (r < 3.5)
    J_exact = erf(np.sqrt(alpha) * r) / np.where(r > 0, r, 1.0)

    err = np.max(np.abs(J[mask] - J_exact[mask]))
    assert err < 5e-4


def test_hartree_is_linear_in_density():
    """J[aρ₁ + bρ₂] = a J[ρ₁] + b J[ρ₂] — convolution is linear."""
    grid = Grid3D(N=32, L=8.0)
    rho1 = _gaussian(grid, alpha=0.8, center=(0.5, 0.0, 0.0))
    rho2 = _gaussian(grid, alpha=1.2, center=(-0.7, 0.3, 0.0))
    a, b = 1.3, -0.4

    J1 = hartree_potential(rho1, grid)
    J2 = hartree_potential(rho2, grid)
    J12 = hartree_potential(a * rho1 + b * rho2, grid)

    err = float(jnp.max(jnp.abs(J12 - (a * J1 + b * J2))))
    assert err < 1e-10


def test_hartree_potential_is_real_and_positive_for_positive_density():
    grid = Grid3D(N=32, L=8.0)
    rho = _gaussian(grid, alpha=1.0)
    J = hartree_potential(rho, grid)
    assert jnp.issubdtype(J.dtype, jnp.floating)
    assert float(jnp.min(J)) > 0.0


def test_hartree_energy_scales_with_alpha():
    """E_H(α) / E_H(α') = √(α/α') for identical normalization."""
    grid = Grid3D(N=48, L=10.0)
    E1 = float(hartree_energy(_gaussian(grid, alpha=0.5), grid))
    E2 = float(hartree_energy(_gaussian(grid, alpha=2.0), grid))
    ratio = E2 / E1
    expected = float(jnp.sqrt(2.0 / 0.5))  # = 2.0
    assert abs(ratio - expected) / expected < 2e-2
