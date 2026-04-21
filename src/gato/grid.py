"""Cartesian grids for 3D Schrödinger solvers.

Conventions
-----------
Atomic units throughout: hbar = m_e = e = 1, lengths in Bohr, energies in Hartree.

Grid layout is *cell-centered* on the cube [-L/2, L/2]^3 with N points per axis
and spacing h = L / N. Grid points sit at

    x_i = -L/2 + (i + 1/2) h,   i = 0, ..., N-1.

No grid point lies on the boundary, which makes both zero-Dirichlet and periodic
boundary conditions symmetric and simple to implement.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Grid3D:
    """Uniform cell-centered 3D Cartesian grid on [-L/2, L/2]^3."""

    N: int
    L: float

    @property
    def h(self) -> float:
        return self.L / self.N

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.N, self.N, self.N)

    @property
    def dV(self) -> float:
        return self.h ** 3

    def axis(self) -> jax.Array:
        """1D coordinate array for one axis, shape (N,)."""
        return -self.L / 2 + (jnp.arange(self.N) + 0.5) * self.h

    def coords(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        """3D coordinate grids X, Y, Z each of shape (N, N, N)."""
        a = self.axis()
        return jnp.meshgrid(a, a, a, indexing="ij")

    def radial(self, softening: float = 0.0) -> jax.Array:
        """sqrt(x² + y² + z² + eps²) at every grid point."""
        X, Y, Z = self.coords()
        return jnp.sqrt(X * X + Y * Y + Z * Z + softening * softening)


def integrate(psi: jax.Array, grid: Grid3D) -> jax.Array:
    """Midpoint-rule integral ∫ ψ dV over the grid domain."""
    return jnp.sum(psi) * grid.dV


def inner_product(phi: jax.Array, psi: jax.Array, grid: Grid3D) -> jax.Array:
    """⟨φ | ψ⟩ = ∫ φ* ψ dV."""
    return jnp.sum(jnp.conj(phi) * psi) * grid.dV


def norm_sq(psi: jax.Array, grid: Grid3D) -> jax.Array:
    """⟨ψ | ψ⟩ = ∫ |ψ|² dV."""
    return jnp.sum(jnp.abs(psi) ** 2) * grid.dV


def normalize(psi: jax.Array, grid: Grid3D) -> jax.Array:
    return psi / jnp.sqrt(norm_sq(psi, grid))
