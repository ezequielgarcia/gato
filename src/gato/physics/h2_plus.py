"""End-to-end Phase 2 driver: the hydrogen molecular ion H₂⁺.

Physics
-------
One electron shared between two protons. In the Born–Oppenheimer
approximation the total energy at fixed nuclear geometry R is

    E(R) = ⟨ψ_el(R) | T̂ + V_ext(R) | ψ_el(R)⟩ + Σ_{i<j} Z_i Z_j / |R_i - R_j|,

where ψ_el is the electronic ground state for the nuclear potential
V_ext(r; R) = -Σ_k Z_k / |r - R_k| (softened). We obtain ψ_el by imaginary-
time propagation in a box.

The Hellmann-Feynman theorem gives nuclear forces as

    F_k = -∂E/∂R_k = -⟨ψ_el | ∂V_ext/∂R_k | ψ_el⟩ - ∂E_nn/∂R_k,

which we compute automatically with `jax.grad` against the energy
*functional* evaluated at the converged ψ_el. No back-propagation through
the imaginary-time solver is needed; variational stationarity of ψ_el
cancels the Pulay contribution.

Exit criterion (from the project README)
----------------------------------------
Pure gradient descent on E(R) should recover the H₂⁺ equilibrium bond
length R_e ≈ 2.00 a₀. The exact (Burrau 1927) value is R_e = 1.9972 a₀
at E = -0.6026 E_h; the softened-Coulomb on a coarse grid typically
reproduces R_e to ~0.05 a₀ and E to ~30 mHa.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from .. import enable_x64
from ..ansatz.grid import init_lcao
from ..geometry import Nuclei, bond_length, nuclear_repulsion, recenter
from ..grid import Grid3D, inner_product, norm_sq, normalize
from ..hamiltonian import Hamiltonian
from ..operators import kinetic
from ..potentials import multi_center_softened_coulomb
from ..solvers import imaginary_time


# ---------------------------------------------------------------------------
# Born-Oppenheimer energy functional
# ---------------------------------------------------------------------------

def bo_energy(
    positions: jax.Array,
    charges: jax.Array,
    psi: jax.Array,
    grid: Grid3D,
    epsilon: float,
    order: int = 4,
) -> jax.Array:
    """Total Born-Oppenheimer energy at fixed electronic wavefunction.

    Returns ⟨ψ|T̂ + V̂(R)|ψ⟩ / ⟨ψ|ψ⟩ + E_nn(R). Differentiable in
    `positions`; `jax.grad(bo_energy, argnums=0)` is the *negative* of the
    Hellmann-Feynman force at converged ψ.
    """
    nuclei = Nuclei(positions=positions, charges=charges)
    V = multi_center_softened_coulomb(nuclei, grid, epsilon)
    T_psi = kinetic(psi, grid.h, "dirichlet", order)
    H_psi = T_psi + V * psi
    n2 = norm_sq(psi, grid).real
    E_el = inner_product(psi, H_psi, grid).real / n2
    E_nn = nuclear_repulsion(nuclei)
    return E_el + E_nn


def hellmann_feynman_force(
    positions: jax.Array,
    charges: jax.Array,
    psi: jax.Array,
    grid: Grid3D,
    epsilon: float,
    order: int = 4,
) -> jax.Array:
    """Force on every nucleus, shape (K, 3). Valid at converged ψ."""
    return -jax.grad(bo_energy, argnums=0)(
        positions, charges, psi, grid, epsilon, order
    )


# ---------------------------------------------------------------------------
# Single-geometry solve
# ---------------------------------------------------------------------------

@dataclass
class SolvePoint:
    nuclei: Nuclei
    psi: jax.Array
    energy: float          # total (electronic + nuclear repulsion)
    electronic_energy: float
    nuclear_repulsion: float


def solve_electronic(
    nuclei: Nuclei,
    grid: Grid3D,
    epsilon: float | None = None,
    n_steps: int = 2000,
    order: int = 4,
) -> SolvePoint:
    """Converge ψ_el at fixed nuclear geometry via imaginary-time propagation."""
    if epsilon is None:
        epsilon = grid.h / 2
    V = multi_center_softened_coulomb(nuclei, grid, epsilon)
    H = Hamiltonian(grid=grid, V=V, boundary="dirichlet", order=order)
    psi0 = init_lcao(nuclei, grid)
    psi0 = normalize(psi0, grid)
    psi, _ = imaginary_time(H, psi0, n_steps=n_steps, log_every=n_steps + 1)
    E_nn = float(nuclear_repulsion(nuclei))
    E_el = float(H.rayleigh(psi))
    return SolvePoint(
        nuclei=nuclei,
        psi=psi,
        energy=E_el + E_nn,
        electronic_energy=E_el,
        nuclear_repulsion=E_nn,
    )


# ---------------------------------------------------------------------------
# Born-Oppenheimer curve
# ---------------------------------------------------------------------------

def bo_curve(
    R_values: list[float] | jax.Array,
    grid: Grid3D,
    Z: float = 1.0,
    epsilon: float | None = None,
    n_steps: int = 2000,
    order: int = 4,
) -> list[SolvePoint]:
    """E(R) for a homonuclear diatomic along the z-axis, atoms at ±R/2."""
    out: list[SolvePoint] = []
    for R in R_values:
        positions = jnp.array([
            [0.0, 0.0, -R / 2],
            [0.0, 0.0, +R / 2],
        ])
        nuclei = Nuclei(positions=positions, charges=jnp.asarray([Z, Z]))
        out.append(solve_electronic(
            nuclei, grid, epsilon=epsilon, n_steps=n_steps, order=order,
        ))
    return out


# ---------------------------------------------------------------------------
# Geometry optimization
# ---------------------------------------------------------------------------

@dataclass
class GeomOptResult:
    final: SolvePoint
    trajectory: list[tuple[int, float, float]]   # (step, energy, bond_length_01)


def optimize_geometry(
    initial_nuclei: Nuclei,
    grid: Grid3D,
    epsilon: float | None = None,
    electronic_steps: int = 2000,
    geom_steps: int = 40,
    learning_rate: float = 0.05,
    order: int = 4,
    verbose: bool = False,
) -> GeomOptResult:
    """Alternating relaxation: solve electrons, take one optax step on R, repeat.

    Pins the charge-weighted COM at the origin after every nuclear update,
    which removes the translational zero-mode without introducing a
    constraint force.
    """
    if epsilon is None:
        epsilon = grid.h / 2
    nuclei = recenter(initial_nuclei)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(nuclei.positions)

    trajectory: list[tuple[int, float, float]] = []
    current = solve_electronic(
        nuclei, grid, epsilon=epsilon, n_steps=electronic_steps, order=order,
    )

    for step in range(geom_steps):
        grad_R = jax.grad(bo_energy, argnums=0)(
            current.nuclei.positions,
            current.nuclei.charges,
            current.psi,
            grid, epsilon, order,
        )
        updates, opt_state = optimizer.update(grad_R, opt_state)
        new_positions = optax.apply_updates(current.nuclei.positions, updates)
        nuclei = recenter(Nuclei(new_positions, current.nuclei.charges))
        current = solve_electronic(
            nuclei, grid, epsilon=epsilon, n_steps=electronic_steps, order=order,
        )
        R_ij = float(bond_length(nuclei, 0, 1)) if nuclei.n >= 2 else float("nan")
        trajectory.append((step, current.energy, R_ij))
        if verbose:
            print(f"  geom step {step:3d}  E = {current.energy:+.6f}  R₀₁ = {R_ij:.4f}")

    return GeomOptResult(final=current, trajectory=trajectory)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Solve H₂⁺ and optimize its geometry.")
    parser.add_argument("--N", type=int, default=48)
    parser.add_argument("--L", type=float, default=10.0)
    parser.add_argument("--R0", type=float, default=2.5,
                        help="Initial H-H separation (a₀).")
    parser.add_argument("--electronic-steps", type=int, default=2000)
    parser.add_argument("--geom-steps", type=int, default=40)
    parser.add_argument("--order", type=int, choices=[2, 4], default=4)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()

    enable_x64()
    grid = Grid3D(N=args.N, L=args.L)
    initial = Nuclei(
        positions=jnp.array([
            [0.0, 0.0, -args.R0 / 2],
            [0.0, 0.0, +args.R0 / 2],
        ]),
        charges=jnp.asarray([1.0, 1.0]),
    )
    print(f"H₂⁺ on a {args.N}³ grid, L = {args.L}, stencil order {args.order}")
    print(f"Initial R = {args.R0}  target R_e ≈ 1.997  (Burrau 1927)")
    result = optimize_geometry(
        initial, grid,
        electronic_steps=args.electronic_steps,
        geom_steps=args.geom_steps,
        learning_rate=args.lr,
        order=args.order,
        verbose=True,
    )
    R_final = float(bond_length(result.final.nuclei, 0, 1))
    print()
    print(f"Final R  : {R_final:.4f}   a₀   (target 1.997)")
    print(f"Final E  : {result.final.energy:+.6f} Ha   (target -0.6026)")


if __name__ == "__main__":
    main()
