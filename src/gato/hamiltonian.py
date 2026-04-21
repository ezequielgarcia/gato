"""Single-particle Hamiltonian H = T + V as a matrix-free linear operator.

The Hamiltonian holds a precomputed potential array V of shape (N, N, N) and
a reference to the grid. Applying H to a wave function costs O(N^3) in both
memory and flops.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .grid import Grid3D, inner_product, norm_sq
from .operators import kinetic


@dataclass(frozen=True)
class Hamiltonian:
    grid: Grid3D
    V: jax.Array
    boundary: str = "dirichlet"
    order: int = 2  # finite-difference order of the kinetic stencil (2 or 4)

    def apply(self, psi: jax.Array) -> jax.Array:
        """Return H ψ = (-½ ∇² + V) ψ."""
        return kinetic(psi, self.grid.h, self.boundary, self.order) + self.V * psi

    def expectation(self, psi: jax.Array) -> jax.Array:
        """⟨ψ|H|ψ⟩ (not normalized). Real for Hermitian H."""
        return inner_product(psi, self.apply(psi), self.grid).real

    def rayleigh(self, psi: jax.Array) -> jax.Array:
        """⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩. Upper bound on E_0."""
        return self.expectation(psi) / norm_sq(psi, self.grid).real
