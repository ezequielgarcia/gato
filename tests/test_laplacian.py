"""Tests for the 2nd-order finite-difference Laplacian.

The key correctness check: plane waves on a periodic grid are exact eigenvectors
of the discrete Laplacian. For e^{i k·r} on a cubic grid with spacing h,

    -∇²_FD ψ = λ_FD ψ,   λ_FD = (2/h²) Σ (1 − cos(k_α h)).

As h → 0, λ_FD → |k|² (the continuum eigenvalue), converging at second order.
"""
import jax
import jax.numpy as jnp

from gato.grid import Grid3D
from gato.operators import gradient, kinetic, laplacian


def _periodic_coords(N: int, L: float):
    h = L / N
    axis = jnp.arange(N) * h
    X, Y, Z = jnp.meshgrid(axis, axis, axis, indexing="ij")
    return X, Y, Z, h


def test_periodic_plane_wave_is_eigenvector():
    N, L = 32, 4.0
    X, Y, Z, h = _periodic_coords(N, L)
    # choose an allowed wave vector k = 2π n / L
    kx, ky, kz = 2 * jnp.pi / L * jnp.array([1.0, 2.0, -1.0])

    psi = jnp.exp(1j * (kx * X + ky * Y + kz * Z))
    lap = laplacian(psi, h, boundary="periodic")

    eig = -2.0 * (
        (1 - jnp.cos(kx * h)) + (1 - jnp.cos(ky * h)) + (1 - jnp.cos(kz * h))
    ) / h**2
    expected = eig * psi
    err = float(jnp.max(jnp.abs(lap - expected)))
    assert err < 1e-10


def test_periodic_continuum_limit_second_order():
    """FD eigenvalue → continuum |k|² at O(h²)."""
    L = 10.0
    k = 2 * jnp.pi / L  # n=1 on each axis
    lam_exact = 3 * k**2  # -∇² eigenvalue (kinetic without the ½)

    errs = []
    for N in (32, 64, 128):
        X, Y, Z, h = _periodic_coords(N, L)
        psi = jnp.exp(1j * k * (X + Y + Z))
        lap = laplacian(psi, h, boundary="periodic")
        # -∇² ψ = λ ψ  →  λ = -lap[0]/psi[0]
        lam_fd = float((-lap[0, 0, 0] / psi[0, 0, 0]).real)
        errs.append(abs(lam_fd - float(lam_exact)))

    # second-order: doubling N (halving h) should reduce error by ~4
    assert errs[1] < errs[0] / 3.5
    assert errs[2] < errs[1] / 3.5


def test_dirichlet_hermitian():
    """⟨φ|∇² ψ⟩ = ⟨∇² φ|ψ⟩ for zero-Dirichlet boundary."""
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    g = Grid3D(N=16, L=5.0)
    phi = jax.random.normal(k1, g.shape, dtype=jnp.float64)
    psi = jax.random.normal(k2, g.shape, dtype=jnp.float64)
    lap_psi = laplacian(psi, g.h, boundary="dirichlet")
    lap_phi = laplacian(phi, g.h, boundary="dirichlet")
    lhs = float(jnp.sum(phi * lap_psi))
    rhs = float(jnp.sum(lap_phi * psi))
    assert abs(lhs - rhs) < 1e-8


def test_dirichlet_particle_in_box_converges():
    """PIB ground state ψ ∝ Π cos(π x/L) on [-L/2, L/2]³ with walls at ±L/2.
    Continuum kinetic energy is 3(π/L)² / 2. The Rayleigh quotient
    ⟨ψ|T̂|ψ⟩ / ⟨ψ|ψ⟩ should converge to it as h → 0.

    Note: the zero-pad Dirichlet convention gives O(h) error at the boundary
    (the analytic cos has nonzero derivative at the wall, so the ghost cell
    value ≈ −π h/(2L) ≠ 0). Globally the Rayleigh quotient still converges,
    but slower than the O(h²) interior stencil would suggest. We verify
    monotonic convergence rather than a strict rate.
    """
    L = 6.0
    E_exact = 3 * (jnp.pi / L) ** 2 / 2

    errs = []
    for N in (32, 64, 128):
        g = Grid3D(N=N, L=L)
        X, Y, Z = g.coords()
        psi = (
            jnp.cos(jnp.pi * X / L)
            * jnp.cos(jnp.pi * Y / L)
            * jnp.cos(jnp.pi * Z / L)
        )
        T_psi = kinetic(psi, g.h, boundary="dirichlet")
        E_fd = float(jnp.sum(psi * T_psi) / jnp.sum(psi * psi))
        errs.append(abs(E_fd - float(E_exact)))

    # monotonic convergence
    assert errs[1] < errs[0]
    assert errs[2] < errs[1]
    # finest grid within 1% of exact
    assert errs[2] / float(E_exact) < 1e-2


def test_gradient_linear_function():
    """∂_x (x) = 1 in the interior."""
    g = Grid3D(N=32, L=6.0)
    X, Y, Z = g.coords()
    grad = gradient(X, g.h, boundary="dirichlet")
    # central region only (avoid the Dirichlet boundary cells)
    s = slice(4, -4)
    assert float(jnp.max(jnp.abs(grad[0, s, s, s] - 1.0))) < 1e-10
    assert float(jnp.max(jnp.abs(grad[1, s, s, s]))) < 1e-10
    assert float(jnp.max(jnp.abs(grad[2, s, s, s]))) < 1e-10


def test_laplacian_invalid_boundary():
    import pytest

    g = Grid3D(N=8, L=2.0)
    psi = jnp.ones(g.shape)
    with pytest.raises(ValueError):
        laplacian(psi, g.h, boundary="mystery")
