"""Raw-grid ansatz: the wave function *is* the parameters.

Parameters are a tensor of shape (N, N, N) of float64 values at grid points.
This is the baseline ansatz; it's what imaginary-time propagation operates on
and the reference against which the neural ansatz is compared.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ..grid import Grid3D


def init_grid_ansatz(
    grid: Grid3D,
    key: jax.Array,
    init: str = "hydrogenic",
    noise: float = 0.0,
) -> jax.Array:
    """Initialize the grid parameters.

    Parameters
    ----------
    init :
        "hydrogenic"  -- ψ₀ = exp(-r), close to the hydrogen 1s ground state.
        "gaussian"    -- ψ₀ = exp(-r²/2).
        "random"      -- unit-normal noise (for debugging).
    noise : additive Gaussian perturbation scale on top of the analytic guess.
    """
    if init == "hydrogenic":
        r = grid.radial(softening=grid.h / 2)
        psi = jnp.exp(-r)
    elif init == "gaussian":
        r = grid.radial()
        psi = jnp.exp(-r * r / 2)
    elif init == "random":
        return jax.random.normal(key, grid.shape)
    else:
        raise ValueError(f"Unknown init {init!r}")

    if noise > 0:
        psi = psi + noise * jax.random.normal(key, grid.shape)
    return psi


def grid_ansatz(params: jax.Array, grid: Grid3D) -> jax.Array:
    """Evaluate the grid ansatz. Trivially the identity."""
    del grid
    return params
