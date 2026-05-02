"""End-to-end Phase 4 driver: the hydrogen molecule H₂ (closed-shell RHF).

The simplest *molecular* RHF: two electrons, two protons, one doubly-occupied
σ_g orbital. Sits between :mod:`gato.physics.h2_plus` (multi-center +
geometry opt, but single electron — no SCF) and
:mod:`gato.physics.water` (multi-center + SCF + HGH pseudopotentials +
geometry opt). Same scf_rhf machinery as :mod:`gato.physics.helium`,
just with a multi-center external potential and nuclei that are allowed
to move.

Why H₂ before H₂O
-----------------
H₂O exercises three new mechanisms simultaneously: pseudopotentials,
many-orbital RHF (n_occ=5), and many-nucleus geometry. H₂ isolates the
"molecular SCF + Hellmann–Feynman forces" combination on the simplest
possible system — same n_occ=1 path validated on helium, but now with
two nuclei and forces on them. No PP machinery, no exchange between
distinct orbitals.

Reference values (RHF complete-basis limit, all-electron)
---------------------------------------------------------
- Bond length R_e:  ~1.40 a₀  (~0.74 Å)        experiment: 0.741 Å
- Total energy:     ~-1.133 E_h  (RHF/CBS)     exact (H₂):  -1.174 E_h
- Atomization:      ~3.6 eV from RHF           experiment:   4.52 eV

The ~40 mHa shortfall vs the exact non-relativistic energy is the
electron-correlation gap that mean-field cannot see by construction —
the same kind of residual seen on helium, recovered by `qato`'s
Slater–Jastrow VMC.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import jax

from .. import enable_x64
from ..geometry import (
    Nuclei,
    bond_length,
    diatomic_along_z,
    nuclear_repulsion,
)
from ..geometry_opt import optimize_geometry_alternating
from ..grid import Grid3D
from ..potentials import multi_center_softened_coulomb
from ..scf import decompose_rhf_energy, scf_rhf


# ---------------------------------------------------------------------------
# Single-geometry RHF solve
# ---------------------------------------------------------------------------

@dataclass
class H2Point:
    nuclei: Nuclei
    orbitals: jax.Array
    energy: float                    # E_RHF + E_nn
    rhf_energy: float
    nuclear_repulsion: float
    n_iters: int
    converged: bool
    orbital_energies: jax.Array


def solve_rhf_at_geometry(
    nuclei: Nuclei,
    grid: Grid3D,
    *,
    epsilon: float | None = None,
    max_iters: int = 30,
    tol: float = 1e-6,
    mixing: float = 0.7,
    order: int = 4,
    lanczos_iters: int = 40,
    initial_orbitals: jax.Array | None = None,
) -> H2Point:
    """RHF on a fixed nuclear configuration with bare (softened) Coulomb."""
    if epsilon is None:
        epsilon = grid.h / 2
    V_ext = multi_center_softened_coulomb(nuclei, grid, epsilon)

    res = scf_rhf(
        V_ext, grid, n_occ=1,
        initial_orbitals=initial_orbitals,
        max_iters=max_iters, tol=tol, mixing=mixing,
        order=order, lanczos_iters=lanczos_iters,
        epsilon=epsilon,
    )

    E_nn = float(nuclear_repulsion(nuclei))
    return H2Point(
        nuclei=nuclei,
        orbitals=res.orbitals,
        energy=float(res.energy) + E_nn,
        rhf_energy=float(res.energy),
        nuclear_repulsion=E_nn,
        n_iters=res.n_iters,
        converged=res.converged,
        orbital_energies=res.orbital_energies,
    )


# ---------------------------------------------------------------------------
# Born–Oppenheimer energy as a function of nuclear positions
# ---------------------------------------------------------------------------

def bo_energy(
    positions: jax.Array,
    charges: jax.Array,
    orbitals: jax.Array,
    grid: Grid3D,
    epsilon: float,
    order: int = 4,
) -> jax.Array:
    """Total BO energy at *fixed* (already converged) orbitals.

    Differentiable in `positions`. Variational stationarity of the SCF
    orbitals at the current geometry means `jax.grad(bo_energy, argnums=0)`
    returns the Hellmann–Feynman force without a Pulay term.
    """
    nuclei = Nuclei(positions=positions, charges=charges)
    V_ext = multi_center_softened_coulomb(nuclei, grid, epsilon)
    terms = decompose_rhf_energy(
        orbitals, V_ext, grid,
        boundary="dirichlet", order=order, epsilon=epsilon,
    )
    return terms.total + nuclear_repulsion(nuclei)


# ---------------------------------------------------------------------------
# Geometry optimization
# ---------------------------------------------------------------------------

@dataclass
class H2GeomOptResult:
    final: H2Point
    trajectory: list[tuple[int, float, float]]   # (step, energy, R_HH)


def optimize_geometry(
    initial_nuclei: Nuclei,
    grid: Grid3D,
    *,
    epsilon: float | None = None,
    geom_steps: int = 30,
    learning_rate: float = 0.05,
    scf_max_iters: int = 30,
    scf_tol: float = 1e-6,
    scf_mixing: float = 0.7,
    order: int = 4,
    verbose: bool = True,
) -> H2GeomOptResult:
    """Alternating SCF + nuclear gradient descent (Adam on positions)."""
    if epsilon is None:
        epsilon = grid.h / 2

    def solve_at(nuclei: Nuclei, prev: H2Point | None) -> H2Point:
        init_orb = prev.orbitals if prev is not None else None
        return solve_rhf_at_geometry(
            nuclei, grid,
            epsilon=epsilon,
            initial_orbitals=init_orb,
            max_iters=scf_max_iters, tol=scf_tol, mixing=scf_mixing, order=order,
        )

    def bo_grad(p: H2Point) -> jax.Array:
        return jax.grad(bo_energy, argnums=0)(
            p.nuclei.positions, p.nuclei.charges, p.orbitals,
            grid, epsilon, order,
        )

    trajectory: list[tuple[int, float, float]] = []

    def on_step(step: int, p: H2Point) -> None:
        R_HH = float(bond_length(p.nuclei, 0, 1))
        trajectory.append((step, p.energy, R_HH))
        if verbose:
            print(
                f"  geom step {step:3d}  E = {p.energy:+.6f}  "
                f"R_HH = {R_HH:.4f} a₀  SCF iters = {p.n_iters}"
            )

    final, _ = optimize_geometry_alternating(
        initial_nuclei,
        solve_at_geometry=solve_at,
        bo_grad=bo_grad,
        geom_steps=geom_steps,
        learning_rate=learning_rate,
        on_step=on_step,
    )
    return H2GeomOptResult(final=final, trajectory=trajectory)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve H₂ with closed-shell RHF and optimize its geometry."
    )
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--L", type=float, default=10.0)
    parser.add_argument("--R0", type=float, default=1.7,
                        help="Initial H–H separation (a₀); R_e ≈ 1.40")
    parser.add_argument("--geom-steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--scf-iters", type=int, default=30)
    parser.add_argument("--mixing", type=float, default=0.7)
    parser.add_argument("--order", type=int, choices=[2, 4], default=4)
    parser.add_argument("--single", action="store_true",
                        help="single-point only, skip geometry optimization")
    args = parser.parse_args()

    enable_x64()
    grid = Grid3D(N=args.N, L=args.L)
    initial = diatomic_along_z(args.R0, Z1=1.0)

    print(f"H₂ on a {args.N}³ grid, L = {args.L}, h = {grid.h:.4f}, stencil order {args.order}")
    print(f"Method: closed-shell RHF, n_occ = 1 (one doubly-occupied σ_g)")
    print(f"Initial R = {args.R0:.3f} a₀   target R_e ≈ 1.40 a₀  (RHF/CBS)")
    print()

    if args.single:
        point = solve_rhf_at_geometry(
            nuclei=initial, grid=grid,
            max_iters=args.scf_iters, mixing=args.mixing, order=args.order,
        )
        print(f"Energy        : {point.energy:+.6f} Ha   (RHF + E_nn)")
        print(f"E_RHF         : {point.rhf_energy:+.6f} Ha")
        print(f"E_nn          : {point.nuclear_repulsion:+.6f} Ha")
        print(f"SCF iters     : {point.n_iters}   converged = {point.converged}")
        print(f"Orbital energy: {float(point.orbital_energies[0]):+.6f} Ha")
        return

    result = optimize_geometry(
        initial, grid,
        geom_steps=args.geom_steps,
        learning_rate=args.lr,
        scf_max_iters=args.scf_iters,
        scf_mixing=args.mixing,
        order=args.order,
        verbose=True,
    )
    R_final = float(bond_length(result.final.nuclei, 0, 1))
    print()
    print(f"Final R_HH    : {R_final:.4f}  a₀  ({R_final * 0.52917721:.3f} Å)")
    print(f"Final energy  : {result.final.energy:+.6f} Ha   (RHF/CBS ≈ -1.133)")


if __name__ == "__main__":
    main()
