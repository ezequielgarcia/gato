"""External potentials as precomputed (N, N, N) arrays on a Grid3D.

Each factory function returns V(r) sampled on the grid. Precomputing the array
(rather than carrying a callable) is efficient because the potential never
changes inside a solver loop, and storing one (N, N, N) array is cheap.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .grid import Grid3D


def softened_coulomb(
    grid: Grid3D,
    Z: float = 1.0,
    epsilon: float | None = None,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> jax.Array:
    """Softened Coulomb potential V(r) = -Z / sqrt(|r - r0|^2 + eps^2).

    Parameters
    ----------
    Z : nuclear charge (Z=1 for hydrogen).
    epsilon : softening scale. Default: h/2 where h is the grid spacing.
        eps=0 is the bare Coulomb singularity and is ill-defined on a grid.
    center : position of the nucleus; defaults to the origin.
    """
    if epsilon is None:
        epsilon = grid.h / 2
    X, Y, Z_coord = grid.coords()
    x0, y0, z0 = center
    r2 = (X - x0) ** 2 + (Y - y0) ** 2 + (Z_coord - z0) ** 2
    return -Z / jnp.sqrt(r2 + epsilon * epsilon)


def harmonic_oscillator(grid: Grid3D, omega: float = 1.0) -> jax.Array:
    """Isotropic 3D harmonic oscillator V(r) = (1/2) omega^2 r^2.

    Ground-state energy is (3/2) * omega in atomic units -- useful as an
    alternative analytic benchmark for the solver.
    """
    X, Y, Z = grid.coords()
    return 0.5 * omega * omega * (X * X + Y * Y + Z * Z)


def constant(grid: Grid3D, v0: float = 0.0) -> jax.Array:
    """Uniform potential v0. Mostly useful as a free-particle baseline."""
    return jnp.full(grid.shape, v0)
