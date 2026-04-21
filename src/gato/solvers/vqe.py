"""Variational Quantum Eigensolver.

Minimize the Rayleigh quotient E[ψ_θ] = ⟨ψ_θ|H|ψ_θ⟩ / ⟨ψ_θ|ψ_θ⟩ over
parameters θ via optax. Works with any ansatz function satisfying

    ansatz(params, grid) -> (N, N, N) array.

Because the loss is scale-invariant in ψ, no explicit normalization step is
required during training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, TypeVar

import jax
import jax.numpy as jnp
import optax

from ..grid import Grid3D
from ..hamiltonian import Hamiltonian

Params = TypeVar("Params")


class Ansatz(Protocol):
    def __call__(self, params: Params, grid: Grid3D) -> jax.Array: ...


@dataclass
class VQEHistory:
    steps: list[int]
    energies: list[float]


def vqe(
    H: Hamiltonian,
    ansatz: Callable[[Params, Grid3D], jax.Array],
    params: Params,
    optimizer: optax.GradientTransformation,
    n_steps: int = 1000,
    log_every: int = 50,
) -> tuple[Params, VQEHistory]:
    """Minimize the Rayleigh quotient E[ψ_θ] by gradient descent."""
    grid = H.grid

    def loss_fn(p):
        psi = ansatz(p, grid)
        Hpsi = H.apply(psi)
        # complex-safe Rayleigh quotient
        num = jnp.sum(jnp.conj(psi) * Hpsi).real
        den = jnp.sum(jnp.abs(psi) ** 2)
        return num / den

    value_and_grad = jax.value_and_grad(loss_fn)

    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state):
        E, grads = value_and_grad(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, E

    steps_hist: list[int] = []
    energies: list[float] = []
    for i in range(n_steps):
        params, opt_state, E = step(params, opt_state)
        if (i % log_every == 0) or (i == n_steps - 1):
            steps_hist.append(i)
            energies.append(float(E))

    return params, VQEHistory(steps=steps_hist, energies=energies)
