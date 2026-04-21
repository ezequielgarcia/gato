"""Sanity tests for Grid3D and integration helpers."""
import jax.numpy as jnp

from gato.grid import Grid3D, inner_product, integrate, norm_sq, normalize


def test_shape_and_spacing():
    g = Grid3D(N=32, L=8.0)
    assert g.shape == (32, 32, 32)
    assert g.axis().shape == (32,)
    assert abs(g.h - 0.25) < 1e-12
    assert abs(g.dV - 0.25**3) < 1e-12


def test_cell_centered_symmetric():
    g = Grid3D(N=10, L=4.0)
    a = g.axis()
    # cell centers are symmetric about zero
    assert float(abs(a[0] + a[-1])) < 1e-12
    # no grid point lies on the boundary
    assert float(abs(a[0])) < g.L / 2
    assert float(abs(a[-1])) < g.L / 2


def test_radial():
    g = Grid3D(N=8, L=4.0)
    r = g.radial()
    assert r.shape == g.shape
    # minimum |r| should be at a corner cell, (h/2)*sqrt(3)
    r_min = float(jnp.min(r))
    assert abs(r_min - (g.h / 2) * jnp.sqrt(3)) < 1e-10


def test_integrate_unit():
    """∫ 1 dV = L³."""
    g = Grid3D(N=16, L=3.0)
    one = jnp.ones(g.shape)
    assert abs(float(integrate(one, g)) - g.L**3) < 1e-10


def test_gaussian_normalization():
    """A unit-norm Gaussian should integrate to 1 on a large enough grid."""
    g = Grid3D(N=64, L=12.0)
    X, Y, Z = g.coords()
    alpha = 1.0
    psi = (alpha / jnp.pi) ** 0.75 * jnp.exp(-alpha * (X**2 + Y**2 + Z**2) / 2)
    assert abs(float(norm_sq(psi, g)) - 1.0) < 1e-3


def test_normalize():
    g = Grid3D(N=32, L=8.0)
    X, Y, Z = g.coords()
    psi = jnp.exp(-(X**2 + Y**2 + Z**2))
    psi_n = normalize(psi, g)
    assert abs(float(norm_sq(psi_n, g)) - 1.0) < 1e-10


def test_inner_product_hermitian():
    """⟨φ|ψ⟩ = ⟨ψ|φ⟩*."""
    g = Grid3D(N=16, L=5.0)
    X, Y, Z = g.coords()
    phi = jnp.exp(-(X**2 + Y**2 + Z**2))
    psi = jnp.exp(-((X - 1) ** 2 + Y**2 + Z**2)) * (1 + 1j * X)
    a = inner_product(phi, psi, g)
    b = inner_product(psi, phi, g)
    assert abs(complex(a) - complex(b).conjugate()) < 1e-10
