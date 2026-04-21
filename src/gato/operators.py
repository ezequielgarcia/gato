"""Differential operators as matrix-free, JIT-compiled functions.

Every operator has signature ``op(psi, h, **kwargs) -> psi``, where ``psi`` is a
3D array of shape (Nx, Ny, Nz) and ``h`` is the uniform grid spacing. This lets
us compose them freely into Hamiltonians without ever materializing an N³ × N³
matrix.

Two Laplacian stencil orders are supported:

- ``order=2`` (default): the classical 7-point stencil, O(h²) accurate.
- ``order=4``: a 13-point stencil using two neighbours along each axis,
  O(h⁴) accurate. Same 3D structure, 1.5× more arithmetic per application,
  but on a smooth wavefunction reaches the same accuracy at half the grid
  size per axis -- roughly 8× cheaper overall.

The 4th-order stencil is derived from

    f''(x) = [-f(x+2h) + 16 f(x+h) - 30 f(x) + 16 f(x-h) - f(x-2h)] / (12 h²)
             + O(h⁴).
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

# Coefficients for the 1D 2nd-order central-difference Laplacian stencil:
# (f[i+1] + f[i-1] - 2 f[i]) / h²   -- each axis contributes independently.
_ORDER2_NEIGHBOUR = 1.0
_ORDER2_CENTER_PER_AXIS = -2.0

# Coefficients for the 1D 4th-order central-difference Laplacian stencil:
# (-f[i+2] + 16 f[i+1] - 30 f[i] + 16 f[i-1] - f[i-2]) / (12 h²)
_ORDER4_NEIGHBOUR_1 = 16.0 / 12.0
_ORDER4_NEIGHBOUR_2 = -1.0 / 12.0
_ORDER4_CENTER_PER_AXIS = -30.0 / 12.0


@partial(jax.jit, static_argnames=("boundary", "order"))
def laplacian(
    psi: jax.Array,
    h: float,
    boundary: str = "dirichlet",
    order: int = 2,
) -> jax.Array:
    """Central-difference 3D Laplacian, matrix-free.

    Parameters
    ----------
    psi : (Nx, Ny, Nz) real or complex array.
    h   : uniform grid spacing.
    boundary :
        "dirichlet" -- ψ is assumed to vanish outside the grid (zero-pad).
        "periodic"  -- ψ wraps (torus topology).
    order :
        2 -- second-order 7-point stencil, O(h²) error.
        4 -- fourth-order 13-point stencil, O(h⁴) error.
    """
    if order == 2:
        return _laplacian_order2(psi, h, boundary)
    if order == 4:
        return _laplacian_order4(psi, h, boundary)
    raise ValueError(f"Unknown order {order!r}; expected 2 or 4.")


def _laplacian_order2(psi, h, boundary):
    if boundary == "periodic":
        acc = 3.0 * _ORDER2_CENTER_PER_AXIS * psi  # -6 ψ
        for ax in range(3):
            acc = acc + jnp.roll(psi, 1, axis=ax) + jnp.roll(psi, -1, axis=ax)
    elif boundary == "dirichlet":
        p = jnp.pad(psi, 1)
        acc = (
            p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1]
            + p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1]
            + p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2]
            + 3.0 * _ORDER2_CENTER_PER_AXIS * p[1:-1, 1:-1, 1:-1]
        )
    else:
        raise ValueError(
            f"Unknown boundary {boundary!r}; expected 'dirichlet' or 'periodic'."
        )
    return acc / (h * h)


def _laplacian_order4(psi, h, boundary):
    c1 = _ORDER4_NEIGHBOUR_1
    c2 = _ORDER4_NEIGHBOUR_2
    center = 3.0 * _ORDER4_CENTER_PER_AXIS  # sum of three axis centers
    if boundary == "periodic":
        acc = center * psi
        for ax in range(3):
            acc = (
                acc
                + c1 * (jnp.roll(psi, 1, axis=ax) + jnp.roll(psi, -1, axis=ax))
                + c2 * (jnp.roll(psi, 2, axis=ax) + jnp.roll(psi, -2, axis=ax))
            )
    elif boundary == "dirichlet":
        # zero-pad two cells on each face
        p = jnp.pad(psi, 2)
        # central cell block: p[2:-2, 2:-2, 2:-2] == psi
        acc = (
            center * p[2:-2, 2:-2, 2:-2]
            # x-axis neighbours
            + c1 * (p[3:-1, 2:-2, 2:-2] + p[1:-3, 2:-2, 2:-2])
            + c2 * (p[4:,    2:-2, 2:-2] + p[:-4,  2:-2, 2:-2])
            # y-axis
            + c1 * (p[2:-2, 3:-1, 2:-2] + p[2:-2, 1:-3, 2:-2])
            + c2 * (p[2:-2, 4:,   2:-2] + p[2:-2, :-4,  2:-2])
            # z-axis
            + c1 * (p[2:-2, 2:-2, 3:-1] + p[2:-2, 2:-2, 1:-3])
            + c2 * (p[2:-2, 2:-2, 4:]   + p[2:-2, 2:-2, :-4])
        )
    else:
        raise ValueError(
            f"Unknown boundary {boundary!r}; expected 'dirichlet' or 'periodic'."
        )
    return acc / (h * h)


@partial(jax.jit, static_argnames=("boundary", "order"))
def kinetic(
    psi: jax.Array,
    h: float,
    boundary: str = "dirichlet",
    order: int = 2,
) -> jax.Array:
    """T̂ ψ = -½ ∇² ψ (atomic units, m_e = ℏ = 1)."""
    return -0.5 * laplacian(psi, h, boundary, order)


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
