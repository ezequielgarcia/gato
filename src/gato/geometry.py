"""Molecular geometry: nuclear positions, charges, bonds, nuclear repulsion.

`Nuclei` is a NamedTuple (a JAX pytree) so that nuclear coordinates flow
naturally through `jax.grad` — nuclear forces are literally `-jax.grad(E)`
with respect to `nuclei.positions`.

Conventions
-----------
Atomic units. Positions are in Bohr, charges are dimensionless nuclear
charges Z_k. `positions` has shape (K, 3), `charges` has shape (K,).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Nuclei(NamedTuple):
    """A set of K nuclei in 3D."""

    positions: jax.Array  # (K, 3) in Bohr
    charges: jax.Array    # (K,)   nuclear charges

    @property
    def n(self) -> int:
        return self.charges.shape[0]


def diatomic_along_z(R: float, Z1: float, Z2: float | None = None) -> Nuclei:
    """Two nuclei on the z-axis, symmetric about the origin: ±R/2.

    Z2 defaults to Z1 (homonuclear). Use this for H₂⁺, H₂, LiH and any
    Born–Oppenheimer curve sweep over a single internuclear distance.
    """
    if Z2 is None:
        Z2 = Z1
    return Nuclei(
        positions=jnp.array([
            [0.0, 0.0, -R / 2],
            [0.0, 0.0, +R / 2],
        ]),
        charges=jnp.asarray([Z1, Z2]),
    )


def bond_length(nuclei: Nuclei, i: int, j: int) -> jax.Array:
    """|R_i - R_j|."""
    return jnp.linalg.norm(nuclei.positions[i] - nuclei.positions[j])


def bond_angle(nuclei: Nuclei, i: int, j: int, k: int) -> jax.Array:
    """Angle (in radians) of the chain i-j-k, vertex at j."""
    a = nuclei.positions[i] - nuclei.positions[j]
    b = nuclei.positions[k] - nuclei.positions[j]
    cos = jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b))
    return jnp.arccos(jnp.clip(cos, -1.0, 1.0))


def nuclear_repulsion(nuclei: Nuclei) -> jax.Array:
    """Classical Coulomb E_nn = Σ_{i<j} Z_i Z_j / |R_i - R_j|.

    Fully differentiable: nuclear forces include the Coulomb term on the
    nuclear skeleton automatically through `jax.grad`.
    """
    K = nuclei.n
    if K < 2:
        return jnp.asarray(0.0, dtype=nuclei.positions.dtype)
    i_idx, j_idx = jnp.triu_indices(K, k=1)
    diffs = nuclei.positions[i_idx] - nuclei.positions[j_idx]
    r_ij = jnp.linalg.norm(diffs, axis=-1)
    Zi = nuclei.charges[i_idx]
    Zj = nuclei.charges[j_idx]
    return jnp.sum(Zi * Zj / r_ij)


def center_of_mass(nuclei: Nuclei) -> jax.Array:
    """Charge-weighted centroid. Used to remove translational drift during
    geometry optimization (the total energy is translation-invariant, so
    `jax.grad` gives zero net force — but optax noise can still induce
    drift over many steps)."""
    return (nuclei.charges[:, None] * nuclei.positions).sum(axis=0) / nuclei.charges.sum()


def recenter(nuclei: Nuclei) -> Nuclei:
    """Shift the molecule so the charge-weighted COM is at the origin."""
    return Nuclei(
        positions=nuclei.positions - center_of_mass(nuclei),
        charges=nuclei.charges,
    )
