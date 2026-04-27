"""End-to-end Phase 3 driver: the helium atom (closed-shell RHF).

Routes through the generic SCF machinery in :mod:`gato.scf`. Mirrors the
shape of :mod:`gato.physics.hydrogen` so that the package layout reflects
the project's phase structure: one named driver per phase, returning a
dataclass with the converged orbital, density, energy decomposition,
and history.

Reference energies (bare Coulomb, complete basis):

- RHF:    −2.862 E_h
- exact:  −2.903 E_h   (correlation gap ≈ 41 mHa to RHF)

The ~41 mHa gap is **the** canonical pedagogical observation: it is exactly
the electron-correlation energy that mean-field cannot see by construction.
The `qato` sibling project recovers it by adding an explicit Jastrow factor
in a Slater–Jastrow VMC trial wavefunction. See IDEAS.md.

On the default softened grid (N=64, L=10, ε=h/2) the solver converges near
−2.6 E_h; the offset to the bare-Coulomb reference is the softening
residual, not an SCF bug. The ε→0 extrapolation closes the gap.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .. import enable_x64
from ..grid import Grid3D
from ..potentials import softened_coulomb
from ..scf import (
    _density,
    decompose_rhf_energy,
    scf_rhf,
)


@dataclass
class HeliumResult:
    grid: Grid3D
    orbitals: jax.Array          # (N, N, N, n_occ) — n_occ=1 for He
    psi: jax.Array               # (N, N, N) — the single doubly-occupied orbital
    density: jax.Array           # (N, N, N) — ρ = 2 |φ|²
    V_ext: jax.Array
    energy: float                # total RHF energy
    kinetic: float               # 2 Σ_i ⟨φ_i|T̂|φ_i⟩
    potential: float             # 2 Σ_i ⟨φ_i|V_ext|φ_i⟩
    hartree: float               # E_H[ρ]
    exchange: float              # E_x
    virial_ratio: float          # 2⟨T⟩ / |⟨V_ext⟩ + E_H + E_x|
    orbital_energies: jax.Array  # (n_occ,)
    n_iters: int
    converged: bool
    history: list[tuple[int, float]]


def solve_helium(
    N: int = 64,
    L: float = 10.0,
    Z: float = 2.0,
    epsilon: float | None = None,
    max_iters: int = 30,
    tol: float = 1e-6,
    mixing: float = 0.7,
    order: int = 4,
    lanczos_iters: int = 40,
    verbose: bool = True,
) -> HeliumResult:
    """Solve closed-shell helium (or He-like Z=3, 4, …) by RHF.

    Parameters
    ----------
    N : grid points per axis.
    L : box side length in Bohr.
    Z : nuclear charge (2 = He, 3 = Li⁺, 4 = Be²⁺, …).
    epsilon : Coulomb softening for both V_ext and the Hartree/exchange
        kernel. Defaults to h/2.
    max_iters, tol, mixing : SCF controls.
    order : finite-difference order of the kinetic stencil (2 or 4).
    lanczos_iters : Krylov dimension for each Fock diagonalization.
    """
    grid = Grid3D(N=N, L=L)
    V_ext = softened_coulomb(grid, Z=Z, epsilon=epsilon)

    result = scf_rhf(
        V_ext, grid, n_occ=1,
        max_iters=max_iters, tol=tol, mixing=mixing,
        order=order, lanczos_iters=lanczos_iters, epsilon=epsilon,
    )

    orbitals = result.orbitals
    rho = _density(orbitals)

    terms = decompose_rhf_energy(
        orbitals, V_ext, grid,
        boundary="dirichlet", order=order, epsilon=epsilon,
    )
    T = float(terms.kinetic)
    V_exp = float(terms.v_ext)
    E_H = float(terms.hartree)
    E_x = float(terms.exchange)

    # Coulomb virial diagnostic: for an exact Coulomb eigenstate the total
    # potential V_total = ⟨V_ext⟩ + ⟨V_ee⟩ satisfies 2⟨T⟩ + ⟨V_total⟩ = 0.
    # For closed-shell RHF on He, ⟨V_ee⟩ = E_H + E_x (the cancellation of the
    # double-count in E_H against the same-orbital exchange J_11 = K_11).
    V_total = V_exp + E_H + E_x
    vr = float(2.0 * T / jnp.abs(V_total))

    history = [(i + 1, e) for i, e in enumerate(result.energy_history)]

    helium = HeliumResult(
        grid=grid,
        orbitals=orbitals,
        psi=orbitals[..., 0],
        density=rho,
        V_ext=V_ext,
        energy=float(result.energy),
        kinetic=T,
        potential=V_exp,
        hartree=E_H,
        exchange=E_x,
        virial_ratio=vr,
        orbital_energies=result.orbital_energies,
        n_iters=result.n_iters,
        converged=result.converged,
        history=history,
    )

    if verbose:
        print(f"Helium-like atom Z={Z} on {N}^3 grid, L = {L}  (h = {grid.h:.4f}, stencil order {order})")
        print(f"Method        : RHF   SCF iters = {result.n_iters}   converged = {result.converged}")
        print(f"Energy        : {helium.energy:+.6f} Ha     (bare-Coulomb RHF ref: -2.862, exact: -2.903)")
        print(f"⟨T⟩           : {T:+.6f} Ha")
        print(f"⟨V_ext⟩       : {V_exp:+.6f} Ha")
        print(f"E_H[ρ]        : {E_H:+.6f} Ha")
        print(f"⟨E_x⟩         : {E_x:+.6f} Ha")
        print(f"ε_HOMO        : {float(result.orbital_energies[0]):+.6f} Ha")
        print(f"Virial 2T/|V| : {vr:+.6f}        (target:  1.000000)")
    return helium


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve the helium-like atom by RHF.")
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--L", type=float, default=10.0)
    parser.add_argument("--Z", type=float, default=2.0, help="Nuclear charge")
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--max-iters", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--mixing", type=float, default=0.7)
    parser.add_argument("--order", type=int, choices=[2, 4], default=4)
    parser.add_argument("--lanczos-iters", type=int, default=40)
    args = parser.parse_args()

    enable_x64()
    solve_helium(
        N=args.N, L=args.L, Z=args.Z, epsilon=args.epsilon,
        max_iters=args.max_iters, tol=args.tol,
        mixing=args.mixing, order=args.order, lanczos_iters=args.lanczos_iters,
    )


if __name__ == "__main__":
    main()
