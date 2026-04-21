"""Neural network trial wave function.

ψ_θ(r) = g_θ(r) · exp(-α_θ |r|),

where g_θ is a small tanh-MLP mapping (x, y, z) -> scalar, and α_θ is a
learnable global length-scale. The exponential envelope imposes the correct
asymptotic decay of a bound state and stabilizes training enormously.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from ..grid import Grid3D


class NeuralAnsatz(eqx.Module):
    """MLP ψ_θ(r) = MLP(r) · exp(-α |r|).

    The MLP has `n_layers` hidden layers of width `hidden` and tanh activations.
    """
    layers: list
    log_alpha: jax.Array   # α = exp(log_alpha), ensures positivity

    def __init__(
        self,
        key: jax.Array,
        hidden: int = 32,
        n_layers: int = 3,
        alpha_init: float = 1.0,
    ):
        keys = jax.random.split(key, n_layers + 1)
        dims = [3] + [hidden] * n_layers + [1]
        self.layers = [
            eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i])
            for i in range(len(dims) - 1)
        ]
        self.log_alpha = jnp.log(jnp.asarray(alpha_init, dtype=jnp.float64))

    def __call__(self, r: jax.Array) -> jax.Array:
        """Evaluate ψ at a single point r = (x, y, z)."""
        h = r
        for lin in self.layers[:-1]:
            h = jnp.tanh(lin(h))
        g = self.layers[-1](h).squeeze()
        alpha = jnp.exp(self.log_alpha)
        return g * jnp.exp(-alpha * jnp.linalg.norm(r))


def neural_ansatz(model: NeuralAnsatz, grid: Grid3D) -> jax.Array:
    """Evaluate a NeuralAnsatz on every grid point, returning (N, N, N)."""
    X, Y, Z = grid.coords()
    coords = jnp.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    psi_flat = jax.vmap(model)(coords)
    return psi_flat.reshape(grid.shape)
