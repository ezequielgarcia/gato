"""Tests for the potentials module."""
import jax.numpy as jnp

from gato.grid import Grid3D
from gato.potentials import harmonic_oscillator, softened_coulomb


def test_softened_coulomb_shape():
    g = Grid3D(N=16, L=8.0)
    V = softened_coulomb(g)
    assert V.shape == g.shape


def test_softened_coulomb_sign_and_asymptote():
    """V < 0 everywhere and V → -1/r far from origin."""
    g = Grid3D(N=32, L=12.0)
    V = softened_coulomb(g, Z=1.0, epsilon=0.1)
    assert float(jnp.max(V)) < 0
    # at a point far from the origin the softened and bare Coulomb agree
    r = g.radial()
    far = r > 3.0
    bare = -1.0 / r
    assert float(jnp.max(jnp.abs(V - bare)[far])) < 1e-3


def test_softened_coulomb_bounded_at_origin():
    g = Grid3D(N=32, L=8.0)
    V = softened_coulomb(g, epsilon=0.2)
    # bounded by -1/eps
    assert float(jnp.min(V)) > -1.0 / 0.2 - 1e-10


def test_softened_coulomb_centered():
    """Potential value at any point equals the value at its antipodal reflection."""
    g = Grid3D(N=16, L=6.0)
    V = softened_coulomb(g)
    flipped = V[::-1, ::-1, ::-1]
    assert float(jnp.max(jnp.abs(V - flipped))) < 1e-12


def test_harmonic_oscillator():
    g = Grid3D(N=16, L=6.0)
    V = harmonic_oscillator(g, omega=1.0)
    X, Y, Z = g.coords()
    expected = 0.5 * (X**2 + Y**2 + Z**2)
    assert float(jnp.max(jnp.abs(V - expected))) < 1e-12
