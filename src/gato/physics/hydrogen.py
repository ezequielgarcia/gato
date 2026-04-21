"""End-to-end Phase 1 driver: the hydrogen atom.

Run with the imaginary-time solver by default (fastest and most reliable for
the grid ansatz). Also exposes a VQE entry point for the neural ansatz.

Expected result: E₀ ≈ -0.5 Hartree on a reasonably converged grid.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from .. import enable_x64
from ..ansatz import grid_ansatz, init_grid_ansatz, neural_ansatz
from ..ansatz.neural import NeuralAnsatz
from ..grid import Grid3D, normalize
from ..hamiltonian import Hamiltonian
from ..observables import (
    kinetic_energy,
    potential_energy,
    virial_ratio,
)
from ..potentials import softened_coulomb
from ..solvers import imaginary_time, vqe


@dataclass
class HydrogenResult:
    grid: Grid3D
    psi: jax.Array
    V: jax.Array
    energy: float
    kinetic: float
    potential: float
    virial_ratio: float
    history: list[tuple[int, float]]


def solve_hydrogen(
    N: int = 48,
    L: float = 10.0,
    epsilon: float | None = None,
    solver: str = "imag_time",
    n_steps: int = 3000,
    seed: int = 0,
    verbose: bool = True,
) -> HydrogenResult:
    """Solve the hydrogen 1s ground state on a cubic grid.

    Parameters
    ----------
    N : grid points per axis.
    L : box side length in Bohr.
    epsilon : Coulomb softening; default h/2.
    solver : "imag_time" or "vqe_neural".
    n_steps : optimizer / propagator steps.
    """
    grid = Grid3D(N=N, L=L)
    V = softened_coulomb(grid, Z=1.0, epsilon=epsilon)
    H = Hamiltonian(grid=grid, V=V, boundary="dirichlet")

    key = jax.random.PRNGKey(seed)

    if solver == "imag_time":
        psi0 = init_grid_ansatz(grid, key, init="hydrogenic")
        psi, history = imaginary_time(H, psi0, n_steps=n_steps, log_every=max(1, n_steps // 20))
        hist = list(zip(history.steps, history.energies))
    elif solver == "vqe_neural":
        # Hydrogen: one nucleus at the origin, Z = 1. The cusp factor
        # exp(-|r|) imposes Kato's exact boundary condition at the nucleus.
        model = NeuralAnsatz(
            key=key,
            nuclei_positions=((0.0, 0.0, 0.0),),
            nuclei_charges=(1.0,),
            hidden=32,
            n_layers=3,
        )
        opt = optax.adam(1e-3)
        model_final, history = vqe(
            H, neural_ansatz, model, opt,
            n_steps=n_steps, log_every=max(1, n_steps // 20),
        )
        psi = neural_ansatz(model_final, grid)
        hist = list(zip(history.steps, history.energies))
    else:
        raise ValueError(f"Unknown solver {solver!r}")

    psi = normalize(psi, grid)

    T = float(kinetic_energy(psi, grid))
    V_exp = float(potential_energy(psi, V, grid))
    E = T + V_exp
    vr = float(virial_ratio(psi, V, grid))

    result = HydrogenResult(
        grid=grid, psi=psi, V=V,
        energy=E, kinetic=T, potential=V_exp,
        virial_ratio=vr, history=hist,
    )

    if verbose:
        print(f"Hydrogen atom on {N}^3 grid, L = {L}  (h = {grid.h:.4f})")
        print(f"Solver        : {solver}   steps = {n_steps}")
        print(f"Energy        : {E:+.6f} Ha     (target: -0.500000)")
        print(f"⟨T⟩           : {T:+.6f} Ha     (analytic: +0.500000)")
        print(f"⟨V⟩           : {V_exp:+.6f} Ha     (analytic: -1.000000)")
        print(f"Virial 2T/|V| : {vr:+.6f}        (target:  1.000000)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve the hydrogen atom.")
    parser.add_argument("--N", type=int, default=48)
    parser.add_argument("--L", type=float, default=10.0)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--solver", choices=["imag_time", "vqe_neural"], default="imag_time")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    enable_x64()
    solve_hydrogen(
        N=args.N, L=args.L, epsilon=args.epsilon,
        solver=args.solver, n_steps=args.steps, seed=args.seed,
    )


if __name__ == "__main__":
    main()
