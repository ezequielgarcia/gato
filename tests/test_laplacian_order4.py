"""Tests for the 4th-order 13-point Laplacian stencil.

Exact dispersion relation on a periodic grid for e^{i k·r}:

    -∇²_FD⁴ ψ = λ^{(4)}_k ψ,
    λ^{(4)}_k = (1/h²) Σ_α [ 2·(15/12) - 2·(16/12) cos(k_α h) + 2·(1/12) cos(2 k_α h) ]
              = (1/h²) Σ_α [ 30/12 - (32/12) cos(k_α h) + (2/12) cos(2 k_α h) ]

As h → 0, λ^{(4)} → |k|² with error O(h⁴) — halving h shrinks the error by ~16,
compared to ~4 for the 2nd-order stencil.
"""
import jax
import jax.numpy as jnp

from gato.grid import Grid3D
from gato.operators import kinetic, laplacian


def _periodic_coords(N, L):
    h = L / N
    axis = jnp.arange(N) * h
    X, Y, Z = jnp.meshgrid(axis, axis, axis, indexing="ij")
    return X, Y, Z, h


def _fd4_eigenvalue(kx, ky, kz, h):
    """Closed form for the 4th-order FD eigenvalue of -∇² on a plane wave."""
    term = (
        (30.0 / 12.0)
        - (32.0 / 12.0) * jnp.cos(kx * h)
        + (2.0 / 12.0) * jnp.cos(2 * kx * h)
    )
    term += (
        (30.0 / 12.0)
        - (32.0 / 12.0) * jnp.cos(ky * h)
        + (2.0 / 12.0) * jnp.cos(2 * ky * h)
    )
    term += (
        (30.0 / 12.0)
        - (32.0 / 12.0) * jnp.cos(kz * h)
        + (2.0 / 12.0) * jnp.cos(2 * kz * h)
    )
    return term / (h * h)


def test_periodic_plane_wave_order4_eigenvalue():
    """e^{ik·r} is an exact eigenvector of the 4th-order Laplacian too."""
    N, L = 32, 4.0
    X, Y, Z, h = _periodic_coords(N, L)
    kx, ky, kz = 2 * jnp.pi / L * jnp.array([1.0, 2.0, -1.0])
    psi = jnp.exp(1j * (kx * X + ky * Y + kz * Z))

    lap = laplacian(psi, h, boundary="periodic", order=4)
    # eigenvalue equation: -∇²_FD⁴ ψ = λ ψ ⇒ lap = -λ ψ
    expected = -_fd4_eigenvalue(kx, ky, kz, h) * psi
    err = float(jnp.max(jnp.abs(lap - expected)))
    assert err < 1e-10


def test_periodic_continuum_limit_fourth_order():
    """FD eigenvalue converges to -|k|² at O(h⁴) — error drops ~16× per
    halving of h, vs. ~4× for the 2nd-order stencil."""
    L = 10.0
    k = 2 * jnp.pi / L
    lam_exact = 3 * k**2

    errs2 = []
    errs4 = []
    for N in (16, 32, 64):
        X, Y, Z, h = _periodic_coords(N, L)
        psi = jnp.exp(1j * k * (X + Y + Z))
        for order, errs in ((2, errs2), (4, errs4)):
            lap = laplacian(psi, h, boundary="periodic", order=order)
            lam_fd = float((-lap[0, 0, 0] / psi[0, 0, 0]).real)
            errs.append(abs(lam_fd - float(lam_exact)))

    # order 2: ratio ~ 4 per halving
    assert errs2[1] < errs2[0] / 3.5
    assert errs2[2] < errs2[1] / 3.5
    # order 4: ratio ~ 16 per halving (relax slightly for finite-precision noise)
    assert errs4[1] < errs4[0] / 12.0
    assert errs4[2] < errs4[1] / 12.0
    # and at matched N, 4th-order error is much smaller than 2nd-order
    assert errs4[-1] < errs2[-1] * 1e-2


def test_order4_dirichlet_hermitian():
    """⟨φ|∇²⁴ ψ⟩ = ⟨∇²⁴ φ|ψ⟩ for random φ, ψ with zero-Dirichlet BC."""
    key = jax.random.PRNGKey(7)
    k1, k2 = jax.random.split(key)
    g = Grid3D(N=16, L=5.0)
    phi = jax.random.normal(k1, g.shape, dtype=jnp.float64)
    psi = jax.random.normal(k2, g.shape, dtype=jnp.float64)
    lap_psi = laplacian(psi, g.h, boundary="dirichlet", order=4)
    lap_phi = laplacian(phi, g.h, boundary="dirichlet", order=4)
    lhs = float(jnp.sum(phi * lap_psi))
    rhs = float(jnp.sum(lap_phi * psi))
    assert abs(lhs - rhs) < 1e-8


def test_order4_kinetic_gaussian():
    """⟨T⟩ of unit-variance 3D Gaussian = 3/4 in atomic units."""
    from gato.grid import inner_product, normalize
    g = Grid3D(N=48, L=10.0)
    X, Y, Z = g.coords()
    psi = normalize(jnp.exp(-(X**2 + Y**2 + Z**2) / 2), g)
    T_psi = kinetic(psi, g.h, boundary="dirichlet", order=4)
    T = float(inner_product(psi, T_psi, g).real)
    assert abs(T - 0.75) < 5e-3


def test_order4_invalid_order_raises():
    import pytest
    g = Grid3D(N=8, L=2.0)
    psi = jnp.ones(g.shape)
    with pytest.raises(ValueError):
        laplacian(psi, g.h, order=3)
    with pytest.raises(ValueError):
        laplacian(psi, g.h, order=6)
