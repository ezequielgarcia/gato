"""Differential operators as matrix-free, JIT-compiled functions.

Every operator has signature ``op(psi, h, **kwargs) -> psi``, where ``psi`` is a
3D array of shape (Nx, Ny, Nz) and ``h`` is the uniform grid spacing. This lets
us compose them freely into Hamiltonians without ever materializing an N³ × N³
matrix.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("boundary",))
def laplacian(psi: jax.Array, h: float, boundary: str = "dirichlet") -> jax.Array:
    """Second-order central-difference 3D Laplacian.

    Parameters
    ----------
    psi : (Nx, Ny, Nz) real or complex array.
    h   : uniform grid spacing.
    boundary :
        "dirichlet" -- ψ is assumed to vanish outside the grid (zero-pad).
        "periodic"  -- ψ wraps (torus topology).
    """
    if boundary == "periodic":
        acc = -6.0 * psi
        for ax in range(3):
            acc = acc + jnp.roll(psi, 1, axis=ax) + jnp.roll(psi, -1, axis=ax)
    elif boundary == "dirichlet":
        p = jnp.pad(psi, 1)
        acc = (
            p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1]
            + p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1]
            + p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2]
            - 6.0 * p[1:-1, 1:-1, 1:-1]
        )
    else:
        raise ValueError(
            f"Unknown boundary {boundary!r}; expected 'dirichlet' or 'periodic'."
        )
    return acc / (h * h)


@partial(jax.jit, static_argnames=("boundary",))
def kinetic(psi: jax.Array, h: float, boundary: str = "dirichlet") -> jax.Array:
    """T̂ ψ = -½ ∇² ψ (atomic units, m_e = ℏ = 1)."""
    return -0.5 * laplacian(psi, h, boundary)


@partial(jax.jit, static_argnames=("boundary",))
def gradient(psi: jax.Array, h: float, boundary: str = "dirichlet") -> jax.Array:
    """Second-order central-difference gradient.

    Returns an array of shape (3, Nx, Ny, Nz) with components (∂x, ∂y, ∂z) ψ.
    """
    if boundary == "periodic":
        comps = [
            (jnp.roll(psi, -1, axis=ax) - jnp.roll(psi, 1, axis=ax)) / (2 * h)
            for ax in range(3)
        ]
    elif boundary == "dirichlet":
        p = jnp.pad(psi, 1)
        comps = [
            (p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) / (2 * h),
            (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1]) / (2 * h),
            (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]) / (2 * h),
        ]
    else:
        raise ValueError(
            f"Unknown boundary {boundary!r}; expected 'dirichlet' or 'periodic'."
        )
    return jnp.stack(comps, axis=0)
