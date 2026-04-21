"""Imaginary-time propagation for the grid ansatz.

In imaginary time, the evolution operator exp(-H τ) projects any state with
nonzero overlap with the ground state onto ψ₀ as τ → ∞. A first-order Euler
discretization is

    ψ ← ψ - Δτ H ψ,    then renormalize.

Stability: Δτ < 1 / λ_max(H). For the 3D FD Laplacian λ_max ≈ 6/h², so any
Δτ < h²/6 is safe. We use Δτ = h²/10 by default.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..grid import norm_sq
from ..hamiltonian import Hamiltonian


@dataclass
class ImagTimeHistory:
    steps: list[int]
    energies: list[float]


def imaginary_time(
    H: Hamiltonian,
    psi0: jax.Array,
    dt: float | None = None,
    n_steps: int = 2000,
    log_every: int = 50,
) -> tuple[jax.Array, ImagTimeHistory]:
    """Propagate ψ₀ in imaginary time until it converges to the ground state."""
    if dt is None:
        dt = H.grid.h ** 2 / 10.0

    @jax.jit
    def step(psi):
        Hpsi = H.apply(psi)
        psi_new = psi - dt * Hpsi
        return psi_new / jnp.sqrt(norm_sq(psi_new, H.grid))

    @jax.jit
    def energy(psi):
        return H.rayleigh(psi)

    psi = psi0 / jnp.sqrt(norm_sq(psi0, H.grid))
    steps_hist: list[int] = []
    energies: list[float] = []

    for i in range(n_steps):
        psi = step(psi)
        if (i % log_every == 0) or (i == n_steps - 1):
            steps_hist.append(i)
            energies.append(float(energy(psi)))

    return psi, ImagTimeHistory(steps=steps_hist, energies=energies)
