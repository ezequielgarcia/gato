"""Hartree potential via FFT convolution on a doubled grid (Hockney method).

The Hartree potential of a density ρ(r) is

    J(r) = ∫ ρ(r') / |r - r'| dV',

which satisfies -∇²J = 4π ρ. On a finite cell-centered grid representing an
*isolated* system, the naive periodic solve Ĵ(k) = 4π ρ̂(k)/|k|² is wrong:
each charge sees its periodic images. The Hockney/Eastwood fix is to
zero-pad ρ into a box of twice the linear size, convolve against a real-space
1/r kernel on the doubled grid, and crop — which eliminates wraparound and
recovers open boundary conditions to within the grid's own discretization error.

The kernel is softened by ε (default h/2) to match the softened external
potential used elsewhere in the package; this keeps the 1/r singularity at
the origin integrable on the grid and ensures ρ and J are consistently
regularized.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from ..grid import Grid3D, integrate


def _coulomb_kernel(N: int, h: float, epsilon: float) -> jax.Array:
    """Softened 1/r kernel on a (2N)^3 grid with r=0 at index (0,0,0).

    Displacements wrap symmetrically: index i ∈ [0, N) represents a positive
    offset i*h; index i ∈ [N, 2N) represents a negative offset (i - 2N)*h.
    This layout is the natural one for circular convolution via FFT and makes
    the convolution of a zero-padded density with this kernel equal to the
    aperiodic integral over the (N*h)^3 support.
    """
    M = 2 * N
    idx = jnp.arange(M)
    disp = jnp.where(idx < N, idx * h, (idx - M) * h)
    X = disp[:, None, None]
    Y = disp[None, :, None]
    Z = disp[None, None, :]
    return 1.0 / jnp.sqrt(X * X + Y * Y + Z * Z + epsilon * epsilon)


@partial(jax.jit, static_argnames=("grid",))
def _hartree_potential_kernel(rho: jax.Array, grid: Grid3D, epsilon: jax.Array) -> jax.Array:
    N, h = grid.N, grid.h
    M = 2 * N
    G = _coulomb_kernel(N, h, epsilon)
    rho_pad = jnp.zeros((M, M, M), dtype=rho.dtype).at[:N, :N, :N].set(rho)
    J_pad = jnp.fft.irfftn(
        jnp.fft.rfftn(rho_pad) * jnp.fft.rfftn(G),
        s=(M, M, M),
    )
    return J_pad[:N, :N, :N] * grid.dV


def hartree_potential(
    rho: jax.Array,
    grid: Grid3D,
    epsilon: float | None = None,
) -> jax.Array:
    """J(r) = ∫ ρ(r') / |r - r'| dV' via doubled-grid FFT convolution.

    Parameters
    ----------
    rho : (N, N, N) real array, the charge (or electron) density.
    grid : the Grid3D rho lives on.
    epsilon : softening scale for the 1/r kernel. Defaults to h/2.

    Returns
    -------
    J : (N, N, N) real array, the Hartree potential on the same grid.
    """
    if epsilon is None:
        epsilon = grid.h / 2
    return _hartree_potential_kernel(rho, grid, epsilon)


@partial(jax.jit, static_argnames=("grid",))
def _hartree_energy_kernel(rho: jax.Array, grid: Grid3D, epsilon: jax.Array) -> jax.Array:
    J = _hartree_potential_kernel(rho, grid, epsilon)
    return 0.5 * integrate(rho * J, grid)


def hartree_energy(rho: jax.Array, grid: Grid3D, epsilon: float | None = None) -> jax.Array:
    """E_H = (1/2) ∫ ρ(r) J(r) dV, the classical electron–electron repulsion."""
    if epsilon is None:
        epsilon = grid.h / 2
    return _hartree_energy_kernel(rho, grid, epsilon)
