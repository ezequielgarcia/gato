"""Neural network trial wave function with exact Kato nuclear cusps.

ψ_θ(r) = g_θ(r) · Π_k exp(-Z_k |r - R_k|),

where g_θ is a tanh-MLP mapping (x, y, z) -> scalar, and the product runs over
nuclei at positions {R_k} with charges {Z_k}. The cusp factor imposes Kato's
exact nuclear-cusp condition

    (1/ψ) ∂ψ/∂r|_{r→R_k}  =  -Z_k

by construction, replacing the learnable exp(-α r) envelope used in the earlier
version of this file (α implicitly rediscovered Z=1 at optimization time for
hydrogen; pinning it to Z makes the constraint exact and generalizes cleanly to
many-center and many-electron systems).

For hydrogen (the default), use `nuclei_positions=((0.0, 0.0, 0.0),)`,
`nuclei_charges=(1.0,)`. For H₂⁺ with bond length R along x̂,
`nuclei_positions=((R/2, 0, 0), (-R/2, 0, 0))`, `nuclei_charges=(1.0, 1.0)`.
"""
from __future__ import annotations

from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

from ..grid import Grid3D


class NeuralAnsatz(eqx.Module):
    """ψ_θ(r) = MLP(r) · Π_k exp(-Z_k |r - R_k|).

    The MLP has `n_layers` hidden tanh layers of width `hidden`. The nuclear
    positions and charges are stored on the module as JAX arrays so they are
    pytree leaves and can be differentiated through in later phases (geometry
    optimization); for Phase 1 the gradient with respect to positions is zero
    by symmetry and they remain at the origin.
    """
    layers: list
    nuclei_positions: jax.Array  # (K, 3)
    nuclei_charges: jax.Array    # (K,)

    def __init__(
        self,
        key: jax.Array,
        nuclei_positions: Sequence[Sequence[float]] = ((0.0, 0.0, 0.0),),
        nuclei_charges: Sequence[float] = (1.0,),
        hidden: int = 32,
        n_layers: int = 3,
    ):
        keys = jax.random.split(key, n_layers + 1)
        dims = [3] + [hidden] * n_layers + [1]
        self.layers = [
            eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i])
            for i in range(len(dims) - 1)
        ]
        self.nuclei_positions = jnp.asarray(nuclei_positions, dtype=jnp.float64)
        self.nuclei_charges = jnp.asarray(nuclei_charges, dtype=jnp.float64)

    def __call__(self, r: jax.Array) -> jax.Array:
        """Evaluate ψ at a single point r = (x, y, z)."""
        h = r
        for lin in self.layers[:-1]:
            h = jnp.tanh(lin(h))
        g = self.layers[-1](h).squeeze()
        # per-nucleus distances: |r - R_k| for each k
        diffs = r - self.nuclei_positions           # (K, 3)
        dists = jnp.linalg.norm(diffs, axis=-1)     # (K,)
        log_cusp = -jnp.sum(self.nuclei_charges * dists)
        return g * jnp.exp(log_cusp)


def neural_ansatz(model: NeuralAnsatz, grid: Grid3D) -> jax.Array:
    """Evaluate a NeuralAnsatz on every grid point, returning (N, N, N)."""
    X, Y, Z = grid.coords()
    coords = jnp.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    psi_flat = jax.vmap(model)(coords)
    return psi_flat.reshape(grid.shape)
