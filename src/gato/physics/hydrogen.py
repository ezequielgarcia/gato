"""End-to-end Phase 1 driver: the hydrogen atom.

Single electron in a softened Coulomb well, solved by imaginary-time
propagation on the 3D Cartesian grid. The simplest possible end-to-end
demonstration that the GATO stack reproduces a known quantum-mechanical
ground state from first principles.

Expected result: E₀ ≈ -0.5 Hartree on a reasonably converged grid.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import jax

from .. import enable_x64
from ..ansatz import init_grid_ansatz
from ..grid import Grid3D, normalize
from ..hamiltonian import Hamiltonian
from ..observables import (
    kinetic_energy,
    potential_energy,
    virial_ratio,
)
from ..potentials import softened_coulomb
from ..solvers import imaginary_time


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
    Z: float = 1.0,
    epsilon: float | None = None,
    n_steps: int = 3000,
    order: int = 2,
    seed: int = 0,
    verbose: bool = True,
) -> HydrogenResult:
    """Solve the hydrogen-like ground state for a one-electron atom with
    nuclear charge Z at the origin.

    Analytic ground state energy: E₀(Z) = -Z²/2 in Hartree.

    Parameters
    ----------
    N : grid points per axis.
    L : box side length in Bohr. Scale with Z: the orbital shrinks as 1/Z.
    Z : nuclear charge (1 = H, 2 = He⁺, 3 = Li²⁺, …).
    epsilon : Coulomb softening; default h/2.
    n_steps : imaginary-time propagator steps.
    order : finite-difference order of the kinetic stencil (2 or 4).
    """
    grid = Grid3D(N=N, L=L)
    V = softened_coulomb(grid, Z=Z, epsilon=epsilon)
    H = Hamiltonian(grid=grid, V=V, boundary="dirichlet", order=order)

    key = jax.random.PRNGKey(seed)
    psi0 = init_grid_ansatz(grid, key, init="hydrogenic", Z=Z)
    psi, history = imaginary_time(H, psi0, n_steps=n_steps, log_every=max(1, n_steps // 20))
    hist = list(zip(history.steps, history.energies))

    psi = normalize(psi, grid)

    T = float(kinetic_energy(psi, grid, order=order))
    V_exp = float(potential_energy(psi, V, grid))
    E = T + V_exp
    vr = float(virial_ratio(psi, V, grid, order=order))

    result = HydrogenResult(
        grid=grid, psi=psi, V=V,
        energy=E, kinetic=T, potential=V_exp,
        virial_ratio=vr, history=hist,
    )

    if verbose:
        E0 = -0.5 * Z * Z
        print(f"Hydrogenic atom Z={Z} on {N}^3 grid, L = {L}  (h = {grid.h:.4f}, stencil order {order})")
        print(f"Solver        : imag_time   steps = {n_steps}")
        print(f"Energy        : {E:+.6f} Ha     (target: {E0:+.6f})")
        print(f"⟨T⟩           : {T:+.6f} Ha     (analytic: {-E0:+.6f})")
        print(f"⟨V⟩           : {V_exp:+.6f} Ha     (analytic: {2*E0:+.6f})")
        print(f"Virial 2T/|V| : {vr:+.6f}        (target:  1.000000)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve the hydrogen-like atom.")
    parser.add_argument("--N", type=int, default=48)
    parser.add_argument("--L", type=float, default=10.0)
    parser.add_argument("--Z", type=float, default=1.0, help="Nuclear charge")
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--order", type=int, choices=[2, 4], default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    enable_x64()
    solve_hydrogen(
        N=args.N, L=args.L, Z=args.Z, epsilon=args.epsilon,
        n_steps=args.steps, order=args.order, seed=args.seed,
    )


if __name__ == "__main__":
    main()
