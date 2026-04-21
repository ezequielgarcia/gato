"""NEPTUNE-Q: differentiable 3D Schrödinger solvers in JAX."""
from __future__ import annotations

import jax

from .grid import Grid3D, inner_product, integrate, norm_sq, normalize
from .hamiltonian import Hamiltonian
from .observables import (
    kinetic_energy,
    potential_energy,
    radial_density,
    virial_ratio,
)
from .operators import gradient, kinetic, laplacian
from .potentials import constant, harmonic_oscillator, softened_coulomb

__all__ = [
    "Grid3D",
    "Hamiltonian",
    "constant",
    "enable_x64",
    "gradient",
    "harmonic_oscillator",
    "inner_product",
    "integrate",
    "kinetic",
    "kinetic_energy",
    "laplacian",
    "main",
    "norm_sq",
    "normalize",
    "potential_energy",
    "radial_density",
    "softened_coulomb",
    "virial_ratio",
]


def enable_x64() -> None:
    """Turn on double precision. Call before any JAX computation."""
    jax.config.update("jax_enable_x64", True)


def main() -> None:
    """Environment smoke test."""
    enable_x64()
    import jax.numpy as jnp

    print(f"jax      : {jax.__version__}")
    print(f"devices  : {jax.devices()}")
    grid = Grid3D(N=32, L=10.0)
    print(f"grid     : N={grid.N}, L={grid.L}, h={grid.h:.4g}, dV={grid.dV:.4g}")
    X, Y, Z = grid.coords()
    psi = jnp.exp(-(X * X + Y * Y + Z * Z) / 2)
    psi = normalize(psi, grid)
    T_psi = kinetic(psi, grid.h)
    energy = float(inner_product(psi, T_psi, grid).real)
    print(f"unit-gaussian KE (analytic 3/4 = 0.75): {energy:.4f}")
