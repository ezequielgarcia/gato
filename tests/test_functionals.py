"""Tests for the LDA exchange and Perdew–Zunger 1981 correlation functionals.

Structure:

1. **Dirac exchange analytics.** On a uniform density ρ = const, the exchange
    energy is $-C_x\\rho^{4/3} V$ in closed form; the potential is pointwise
    $-\\tfrac{4}{3}C_x\\rho^{1/3}$. Checked to machine precision.

2. **Variational consistency via `jax.grad`.** For any functional of ρ written
    as ``E[ρ] = Σ f(ρ_i) dV``, the discrete functional derivative is
    ``dE/dρ_i = f'(ρ_i) dV``. Dividing out ``dV`` gives the pointwise potential.
    Checking ``V_x`` and ``V_c`` this way against `jax.grad(E)(rho)/dV` catches
    any algebraic slip in the hand-derived expressions for ``dE/dρ``.

3. **PZ81 branch continuity at $r_s = 1$.** The Perdew–Zunger parameters are
    fit so that both $\\epsilon_c$ and its derivative (hence $V_c$) match at
    the switching point between the high- and low-density forms; we check
    this explicitly because a mismatch would produce a discontinuous
    potential at the transition, which would break SCF convergence in
    systems spanning both regimes.
"""
import jax
import jax.numpy as jnp
import numpy as np

from gato.functionals import (
    lda_correlation_energy,
    lda_correlation_energy_density,
    lda_correlation_potential,
    lda_exchange_energy,
    lda_exchange_energy_density,
    lda_exchange_potential,
    lda_xc_energy,
    lda_xc_potential,
)
from gato.grid import Grid3D


_CX = 0.75 * (3.0 / float(jnp.pi)) ** (1.0 / 3.0)


def test_exchange_uniform_density_matches_dirac():
    grid = Grid3D(N=16, L=4.0)
    rho0 = 0.5
    rho = jnp.full(grid.shape, rho0)

    E = float(lda_exchange_energy(rho, grid))
    V = grid.L ** 3
    expected = -_CX * rho0 ** (4.0 / 3.0) * V
    assert abs(E - expected) / abs(expected) < 1e-12


def test_exchange_potential_pointwise_formula():
    rho = jnp.linspace(1e-3, 2.0, 50)
    V = lda_exchange_potential(rho)
    expected = -(4.0 / 3.0) * _CX * rho ** (1.0 / 3.0)
    err = float(jnp.max(jnp.abs(V - expected)))
    assert err < 1e-14


def test_exchange_potential_is_variational_derivative():
    """V_x = δE_x/δρ. On the grid, d E_x / d ρ_i = V_x(r_i) · dV."""
    grid = Grid3D(N=8, L=2.0)
    key = jax.random.PRNGKey(0)
    rho = 0.3 + 0.1 * jax.random.uniform(key, grid.shape)

    grad_E = jax.grad(lda_exchange_energy)(rho, grid)
    V_num = grad_E / grid.dV
    V_an = lda_exchange_potential(rho)

    err = float(jnp.max(jnp.abs(V_num - V_an)))
    assert err < 1e-12


def test_correlation_potential_is_variational_derivative():
    """Same δE/δρ check for the PZ81 correlation functional, over a density
    range that straddles r_s = 1 so both branches are exercised."""
    grid = Grid3D(N=8, L=2.0)
    key = jax.random.PRNGKey(1)
    # mix of high-density (rs<1, ρ > 3/(4π) ≈ 0.24) and low-density regions
    rho = 0.05 + 0.5 * jax.random.uniform(key, grid.shape)

    grad_E = jax.grad(lda_correlation_energy)(rho, grid)
    V_num = grad_E / grid.dV
    V_an = lda_correlation_potential(rho)

    err = float(jnp.max(jnp.abs(V_num - V_an)))
    assert err < 1e-10


def test_pz81_continuity_at_rs_one():
    """Branch-matching at r_s = 1.

    PZ81 publishes parameters that enforce continuity of ε_c and V_c at
    r_s = 1 only to the precision of the rounded coefficients (~3·10⁻⁵ Ha
    for ε_c, similar for V_c). This is the published parameter set used in
    QE, ABINIT and other production codes, so we check the mismatch is at
    that level rather than insisting on machine precision.
    """
    rho_star = 3.0 / (4.0 * np.pi)
    rho = jnp.asarray([rho_star * (1 + 1e-10), rho_star * (1 - 1e-10)])

    eps = lda_correlation_energy_density(rho)
    V = lda_correlation_potential(rho)
    assert abs(float(eps[0] - eps[1])) < 1e-4
    assert abs(float(V[0] - V[1])) < 1e-4


def test_xc_is_sum_of_components():
    grid = Grid3D(N=12, L=3.0)
    key = jax.random.PRNGKey(2)
    rho = 0.1 + 0.3 * jax.random.uniform(key, grid.shape)

    E_xc = float(lda_xc_energy(rho, grid))
    E_x = float(lda_exchange_energy(rho, grid))
    E_c = float(lda_correlation_energy(rho, grid))
    assert abs(E_xc - (E_x + E_c)) < 1e-12

    V_xc = lda_xc_potential(rho)
    V_x = lda_exchange_potential(rho)
    V_c = lda_correlation_potential(rho)
    err = float(jnp.max(jnp.abs(V_xc - (V_x + V_c))))
    assert err < 1e-14


def test_exchange_energy_scales_with_rho_4_3():
    """E_x[λρ] = λ^(4/3) E_x[ρ] for the Dirac functional."""
    grid = Grid3D(N=16, L=4.0)
    key = jax.random.PRNGKey(3)
    rho = 0.1 + 0.3 * jax.random.uniform(key, grid.shape)
    E1 = float(lda_exchange_energy(rho, grid))
    lam = 2.5
    E2 = float(lda_exchange_energy(lam * rho, grid))
    assert abs(E2 - lam ** (4.0 / 3.0) * E1) / abs(E1) < 1e-12
