"""The multi-center softened Coulomb potential."""
import jax
import jax.numpy as jnp
import pytest

from gato.geometry import Nuclei
from gato.grid import Grid3D
from gato.potentials import multi_center_softened_coulomb, softened_coulomb


def test_reduces_to_single_center_when_K_equals_1():
    """One nucleus at the origin: result must equal `softened_coulomb`."""
    grid = Grid3D(N=24, L=8.0)
    eps = 0.3
    nuclei = Nuclei(
        positions=jnp.zeros((1, 3)),
        charges=jnp.asarray([1.0]),
    )
    V_multi = multi_center_softened_coulomb(nuclei, grid, epsilon=eps)
    V_single = softened_coulomb(grid, Z=1.0, epsilon=eps, center=(0.0, 0.0, 0.0))
    assert jnp.allclose(V_multi, V_single, atol=1e-14)


def test_charge_scaling():
    """V scales linearly with Z when the geometry is fixed."""
    grid = Grid3D(N=20, L=8.0)
    eps = 0.3
    R = jnp.array([[0.0, 0.0, 0.0]])
    V1 = multi_center_softened_coulomb(
        Nuclei(positions=R, charges=jnp.asarray([1.0])), grid, epsilon=eps,
    )
    V3 = multi_center_softened_coulomb(
        Nuclei(positions=R, charges=jnp.asarray([3.0])), grid, epsilon=eps,
    )
    assert jnp.allclose(V3, 3.0 * V1, rtol=1e-12)


def test_two_centers_superposition():
    """Sum of two identical copies at ±z equals the one-call result for both."""
    grid = Grid3D(N=32, L=10.0)
    eps = 0.3
    a, b = jnp.array([0.0, 0.0, -1.0]), jnp.array([0.0, 0.0, 1.0])

    Va = multi_center_softened_coulomb(
        Nuclei(positions=a[None, :], charges=jnp.asarray([1.0])), grid, epsilon=eps,
    )
    Vb = multi_center_softened_coulomb(
        Nuclei(positions=b[None, :], charges=jnp.asarray([1.0])), grid, epsilon=eps,
    )
    V_both = multi_center_softened_coulomb(
        Nuclei(positions=jnp.stack([a, b]), charges=jnp.asarray([1.0, 1.0])),
        grid, epsilon=eps,
    )
    assert jnp.allclose(V_both, Va + Vb, atol=1e-12)


def test_potential_is_differentiable_in_positions():
    """`jax.grad` must flow through the potential to the nuclear positions.
    This is the prerequisite for every force computation in Phase 2+."""
    grid = Grid3D(N=16, L=6.0)
    charges = jnp.asarray([1.0, 1.0])

    def scalar_of_positions(positions):
        nuclei = Nuclei(positions=positions, charges=charges)
        V = multi_center_softened_coulomb(nuclei, grid, epsilon=0.4)
        return V.sum()

    positions = jnp.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    grad_R = jax.grad(scalar_of_positions)(positions)
    assert grad_R.shape == (2, 3)
    assert jnp.all(jnp.isfinite(grad_R))
    # Homonuclear z-axis: ∂_z on atom 0 must equal -∂_z on atom 1 by parity.
    assert grad_R[0, 2] == pytest.approx(-grad_R[1, 2], rel=1e-10)
