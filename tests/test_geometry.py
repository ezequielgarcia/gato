"""Nuclei, bond observables, and nuclear repulsion."""
import jax
import jax.numpy as jnp
import pytest

from gato.geometry import (
    Nuclei,
    bond_angle,
    bond_length,
    center_of_mass,
    nuclear_repulsion,
    recenter,
)


def _h2(R: float) -> Nuclei:
    return Nuclei(
        positions=jnp.array([[0.0, 0.0, -R / 2], [0.0, 0.0, R / 2]]),
        charges=jnp.asarray([1.0, 1.0]),
    )


def test_bond_length_simple():
    nuclei = _h2(2.0)
    assert float(bond_length(nuclei, 0, 1)) == pytest.approx(2.0)
    # Symmetric.
    assert float(bond_length(nuclei, 1, 0)) == pytest.approx(2.0)


def test_bond_angle_right_and_straight():
    # Three nuclei at (1,0,0), (0,0,0), (0,1,0) → 90° at the middle atom.
    nuclei = Nuclei(
        positions=jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        charges=jnp.asarray([1.0, 1.0, 1.0]),
    )
    theta = float(bond_angle(nuclei, 0, 1, 2))
    assert theta == pytest.approx(jnp.pi / 2, abs=1e-6)
    # Colinear: atoms at -1, 0, +1 on z → 180°.
    colinear = Nuclei(
        positions=jnp.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        charges=jnp.asarray([1.0, 1.0, 1.0]),
    )
    assert float(bond_angle(colinear, 0, 1, 2)) == pytest.approx(jnp.pi, abs=1e-6)


def test_nuclear_repulsion_h2():
    """E_nn for two unit charges at separation R is 1/R."""
    for R in (1.0, 2.0, 3.5):
        nuclei = _h2(R)
        assert float(nuclear_repulsion(nuclei)) == pytest.approx(1.0 / R, rel=1e-10)


def test_nuclear_repulsion_vanishes_for_single_atom():
    single = Nuclei(positions=jnp.zeros((1, 3)), charges=jnp.asarray([1.0]))
    assert float(nuclear_repulsion(single)) == pytest.approx(0.0, abs=1e-14)


def test_nuclei_is_a_jax_pytree():
    """Crucial for `jax.grad` to flow through positions. NamedTuples are
    pytrees by construction, but we verify explicitly."""
    nuclei = _h2(2.0)
    leaves, _ = jax.tree_util.tree_flatten(nuclei)
    assert len(leaves) == 2   # positions, charges


def test_forces_from_nuclear_repulsion_match_analytic():
    """For two unit charges on the z-axis at ±R/2: F_z on atom 0 = -1/R²,
    F_z on atom 1 = +1/R². `jax.grad` must deliver this exactly."""
    R = 2.0
    positions = jnp.array([[0.0, 0.0, -R / 2], [0.0, 0.0, R / 2]])
    charges = jnp.asarray([1.0, 1.0])

    def E_nn(pos):
        return nuclear_repulsion(Nuclei(positions=pos, charges=charges))

    force = -jax.grad(E_nn)(positions)
    assert force[0, 2] == pytest.approx(-1.0 / (R * R), rel=1e-12)
    assert force[1, 2] == pytest.approx(+1.0 / (R * R), rel=1e-12)
    # No force in x, y.
    assert jnp.allclose(force[:, :2], 0.0, atol=1e-14)


def test_center_of_mass_and_recenter():
    off = Nuclei(
        positions=jnp.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        charges=jnp.asarray([1.0, 1.0]),
    )
    com = center_of_mass(off)
    assert jnp.allclose(com, jnp.array([2.0, 0.0, 0.0]))
    centered = recenter(off)
    assert jnp.allclose(center_of_mass(centered), 0.0, atol=1e-14)
