"""Observables: kinetic / potential / total energy, virial ratio, radial density.

All observables take a wave function and divide by ⟨ψ|ψ⟩, so the caller need
not pre-normalize.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .grid import Grid3D, inner_product, norm_sq
from .operators import kinetic


def kinetic_energy(
    psi: jax.Array, grid: Grid3D, boundary: str = "dirichlet"
) -> jax.Array:
    """⟨T̂⟩ = ⟨ψ|(-½ ∇²)|ψ⟩ / ⟨ψ|ψ⟩."""
    T_psi = kinetic(psi, grid.h, boundary)
    return inner_product(psi, T_psi, grid).real / norm_sq(psi, grid).real


def potential_energy(psi: jax.Array, V: jax.Array, grid: Grid3D) -> jax.Array:
    """⟨V̂⟩ = ⟨ψ|V|ψ⟩ / ⟨ψ|ψ⟩ for multiplicative V(r)."""
    Vpsi = V * psi
    return inner_product(psi, Vpsi, grid).real / norm_sq(psi, grid).real


def virial_ratio(
    psi: jax.Array, V: jax.Array, grid: Grid3D, boundary: str = "dirichlet"
) -> jax.Array:
    """Return 2⟨T⟩ / |⟨V⟩|.

    For an exact Coulomb eigenstate the virial theorem gives 2⟨T⟩ + ⟨V⟩ = 0,
    i.e., this ratio is +1.0 (because ⟨V⟩ < 0 for Coulomb). Deviations
    measure how non-eigenstate the trial ψ is.
    """
    T = kinetic_energy(psi, grid, boundary)
    Vv = potential_energy(psi, V, grid)
    return 2 * T / jnp.abs(Vv)


def radial_density(
    psi: jax.Array, grid: Grid3D, n_bins: int = 100, r_max: float | None = None
) -> tuple[jax.Array, jax.Array]:
    """Radial probability density P(r) with ∫ P(r) dr = ⟨ψ|ψ⟩.

    Returns (r_centers, P). The analytic hydrogen 1s is P(r) = 4 r² exp(-2 r).
    """
    r = grid.radial().reshape(-1)
    density = (jnp.abs(psi) ** 2 * grid.dV).reshape(-1)
    if r_max is None:
        r_max = float(grid.L / 2)
    bins = jnp.linspace(0.0, r_max, n_bins + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])
    widths = bins[1:] - bins[:-1]
    hist, _ = jnp.histogram(r, bins=bins, weights=density)
    return centers, hist / widths
