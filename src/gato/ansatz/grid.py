"""Raw-grid ansatz: the wave function *is* the parameters.

Parameters are a tensor of shape (N, N, N) of float64 values at grid points.
This is the baseline ansatz; it's what imaginary-time propagation operates on
and the reference against which the neural ansatz is compared.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ..geometry import Nuclei
from ..grid import Grid3D


def init_grid_ansatz(
    grid: Grid3D,
    key: jax.Array,
    init: str = "hydrogenic",
    Z: float = 1.0,
    noise: float = 0.0,
) -> jax.Array:
    """Initialize the grid parameters.

    Parameters
    ----------
    init :
        "hydrogenic"  -- ψ₀ = exp(-Z r), the exact hydrogen-like 1s shape.
        "gaussian"    -- ψ₀ = exp(-r²/2).
        "random"      -- unit-normal noise (for debugging).
    Z : nuclear charge (only used for init="hydrogenic").
    noise : additive Gaussian perturbation scale on top of the analytic guess.
    """
    if init == "hydrogenic":
        r = grid.radial(softening=grid.h / 2)
        psi = jnp.exp(-Z * r)
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


def init_lcao(nuclei: Nuclei, grid: Grid3D, softening: float | None = None) -> jax.Array:
    """LCAO initial guess: sum of hydrogenic 1s at each nucleus.

    ψ₀(r) = Σ_k Z_k^{3/2} exp(-Z_k |r - R_k|), unnormalized.

    This is the bonding σ_g combination for a homonuclear diatomic at any
    separation (both coefficients are +1). For heteronuclear systems the
    charge-weighting gives reasonable starting amplitudes; imag-time then
    relaxes to the true ground state.
    """
    if softening is None:
        softening = grid.h / 2
    X, Y, Z = grid.coords()
    grid_pts = jnp.stack([X, Y, Z], axis=-1)

    def one_center(R, Zk):
        diff = grid_pts - R
        r = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + softening * softening)
        return (Zk ** 1.5) * jnp.exp(-Zk * r)

    contribs = jax.vmap(one_center)(nuclei.positions, nuclei.charges)
    return jnp.sum(contribs, axis=0)
