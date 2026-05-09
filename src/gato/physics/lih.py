"""End-to-end Phase 4 driver: lithium hydride LiH (closed-shell RHF + HGH PP).

The simplest *heteronuclear* molecule, and the simplest molecular use of
HGH pseudopotentials. Two atoms (Li, H), two valence electrons after the
Li 1s² core is absorbed into the pseudopotential, one doubly-occupied σ
orbital. Cost class is identical to H₂ ($n_\\text{occ}=1$); the new
machinery exercised is the HGH non-local projector path on Li, which
H₂'s all-Coulomb run does not touch.

Pedagogical role
----------------
LiH sits between :mod:`gato.physics.h2` (homonuclear, no PP, n_occ=1) and
:mod:`gato.physics.water` (heteronuclear, PP, n_occ=4). It is the
cheapest test that the **HGH non-local + multi-center geometry-opt
combination actually works** in a molecular setting. The pedagogical
payoff is the dipole moment: Li gives up its 2s electron almost
completely to H, producing one of the largest dipole-per-bond ratios in
chemistry.

Reference values (RHF complete-basis limit with HGH-LDA pseudopotentials)
-------------------------------------------------------------------------
- Bond length R_e:  ~3.02 a₀  (~1.60 Å)        experiment: 1.595 Å
- Total energy:     ~-0.78 E_h with HGH        (PP-dependent absolute)
- Dipole moment:    ~6.0 D                      experiment: 5.88 D

The dipole — large, positive on Li, pointing Li⁺→H⁻ — is the demo:
chemistry's classical "ionic bond" picture is recovered from
Schrödinger + Pauli + minimization on a nominally neutral molecule.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .. import enable_x64
from ..geometry import (
    Nuclei,
    bond_length,
    diatomic_along_z,
    nuclear_repulsion,
)
from ..geometry_opt import optimize_geometry_alternating
from ..grid import Grid3D
from ..pseudopotentials import (
    HGHParams,
    hgh_local_potential,
    hgh_nonlocal_apply,
    lookup,
    total_valence_charge,
)
from ..scf import decompose_rhf_energy, scf_rhf


# ---------------------------------------------------------------------------
# Initial geometry
# ---------------------------------------------------------------------------

def lih_initial_geometry(R_LiH: float = 3.3) -> Nuclei:
    """Li at -R/2, H at +R/2 along z. Default starts a bit longer than R_e
    (~3.02 a₀) so geometry optimization has a non-trivial trajectory."""
    return diatomic_along_z(R_LiH, Z1=3.0, Z2=1.0)


def lih_hgh_elements() -> tuple[HGHParams, ...]:
    """HGH parameter list aligned with `lih_initial_geometry`."""
    return (lookup("Li"), lookup("H"))


# ---------------------------------------------------------------------------
# Pseudopotential nuclear repulsion (re-used from water; DRY-mirrored here
# for clarity, since lih.py is meant to be readable in isolation).
# ---------------------------------------------------------------------------

def pp_nuclear_repulsion(
    nuclei: Nuclei,
    elements: tuple[HGHParams, ...],
) -> jax.Array:
    """E_nn for pseudoionic nuclei: Σ_{i<j} Z_ion_i Z_ion_j / |R_i - R_j|.

    Uses the *valence* charges (Z_ion), not the bare nuclear charges, since
    each ion's long-range tail is screened by its frozen core.
    """
    K = nuclei.n
    if K < 2:
        return jnp.asarray(0.0, dtype=nuclei.positions.dtype)
    Z_ion = jnp.asarray([p.Z_ion for p in elements])
    pp_nuclei = Nuclei(positions=nuclei.positions, charges=Z_ion)
    return nuclear_repulsion(pp_nuclei)


# ---------------------------------------------------------------------------
# Single-geometry RHF solve
# ---------------------------------------------------------------------------

@dataclass
class LiHPoint:
    nuclei: Nuclei
    orbitals: jax.Array
    energy: float
    rhf_energy: float
    nuclear_repulsion: float
    n_iters: int
    converged: bool
    orbital_energies: jax.Array


def solve_rhf_at_geometry(
    nuclei: Nuclei,
    elements: tuple[HGHParams, ...],
    grid: Grid3D,
    *,
    max_iters: int = 50,
    tol: float = 1e-6,
    mixing: float = 0.5,
    order: int = 4,
    lanczos_iters: int = 60,
    initial_orbitals: jax.Array | None = None,
) -> LiHPoint:
    """RHF on a fixed nuclear configuration with HGH pseudopotentials."""
    V_local = hgh_local_potential(nuclei, elements, grid)

    def V_nl(psi: jax.Array) -> jax.Array:
        return hgh_nonlocal_apply(psi, nuclei, elements, grid)

    n_electrons = int(total_valence_charge(elements))
    if n_electrons % 2 != 0:
        raise ValueError(
            f"Closed-shell RHF requires even electron count; got {n_electrons}"
        )
    n_occ = n_electrons // 2

    res = scf_rhf(
        V_local, grid, n_occ=n_occ,
        initial_orbitals=initial_orbitals,
        max_iters=max_iters, tol=tol, mixing=mixing,
        order=order, lanczos_iters=lanczos_iters,
        V_nl_apply=V_nl,
    )

    E_nn = float(pp_nuclear_repulsion(nuclei, elements))
    return LiHPoint(
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
    elements: tuple[HGHParams, ...],
    grid: Grid3D,
    order: int = 4,
) -> jax.Array:
    """Total BO energy at fixed orbitals; differentiable in `positions`."""
    nuclei = Nuclei(positions=positions, charges=charges)
    V_local = hgh_local_potential(nuclei, elements, grid)

    def V_nl(psi: jax.Array) -> jax.Array:
        return hgh_nonlocal_apply(psi, nuclei, elements, grid)

    terms = decompose_rhf_energy(
        orbitals, V_local, grid,
        boundary="dirichlet", order=order,
        epsilon=None, V_nl_apply=V_nl,
    )
    return terms.total + pp_nuclear_repulsion(nuclei, elements)


# Module-level cached force function. argnums (3, 4, 5) = (elements, grid, order)
# are static; positions/charges/orbitals are traced. Caching here avoids
# re-tracing on every geometry step.
_bo_grad_positions = jax.jit(
    jax.grad(bo_energy, argnums=0),
    static_argnums=(3, 4, 5),
)


# ---------------------------------------------------------------------------
# Geometry optimization
# ---------------------------------------------------------------------------

@dataclass
class LiHGeomOptResult:
    final: LiHPoint
    trajectory: list[tuple[int, float, float]]   # (step, energy, R_LiH)


def optimize_geometry(
    initial_nuclei: Nuclei,
    elements: tuple[HGHParams, ...],
    grid: Grid3D,
    *,
    geom_steps: int = 30,
    learning_rate: float = 0.05,
    scf_max_iters: int = 50,
    scf_tol: float = 1e-6,
    scf_mixing: float = 0.5,
    order: int = 4,
    verbose: bool = True,
) -> LiHGeomOptResult:
    """Alternating SCF + nuclear gradient descent (Adam on positions)."""
    def solve_at(nuclei: Nuclei, prev: LiHPoint | None) -> LiHPoint:
        init_orb = prev.orbitals if prev is not None else None
        return solve_rhf_at_geometry(
            nuclei, elements, grid,
            initial_orbitals=init_orb,
            max_iters=scf_max_iters, tol=scf_tol, mixing=scf_mixing, order=order,
        )

    def bo_grad(p: LiHPoint) -> jax.Array:
        return _bo_grad_positions(
            p.nuclei.positions, p.nuclei.charges, p.orbitals,
            elements, grid, order,
        )

    trajectory: list[tuple[int, float, float]] = []

    def on_step(step: int, p: LiHPoint) -> None:
        R = float(bond_length(p.nuclei, 0, 1))
        trajectory.append((step, p.energy, R))
        if verbose:
            print(
                f"  geom step {step:3d}  E = {p.energy:+.6f}  "
                f"R_LiH = {R:.4f} a₀  SCF iters = {p.n_iters}"
            )

    final, _ = optimize_geometry_alternating(
        initial_nuclei,
        solve_at_geometry=solve_at,
        bo_grad=bo_grad,
        geom_steps=geom_steps,
        learning_rate=learning_rate,
        on_step=on_step,
    )
    return LiHGeomOptResult(final=final, trajectory=trajectory)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve LiH with RHF + HGH pseudopotentials and optimize geometry."
    )
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--L", type=float, default=14.0)
    parser.add_argument("--R0", type=float, default=3.3,
                        help="initial Li-H distance in Bohr (R_e ≈ 3.02)")
    parser.add_argument("--geom-steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--scf-iters", type=int, default=50)
    parser.add_argument("--mixing", type=float, default=0.5)
    parser.add_argument("--order", type=int, choices=[2, 4], default=4)
    parser.add_argument("--single", action="store_true",
                        help="single-point only, skip geometry optimization")
    args = parser.parse_args()

    enable_x64()
    grid = Grid3D(N=args.N, L=args.L)
    nuclei = lih_initial_geometry(R_LiH=args.R0)
    elements = lih_hgh_elements()

    print(f"LiH on a {args.N}³ grid, L = {args.L}, h = {grid.h:.4f}, stencil order {args.order}")
    print(f"Pseudopotentials: HGH-LDA-q1 Li (Z_ion=1) + HGH H (Z_ion=1); 2 valence electrons")
    print(f"Initial: R_LiH = {args.R0:.3f} a₀     target R_e ≈ 3.02 a₀ (1.60 Å)")
    print()

    if args.single:
        point = solve_rhf_at_geometry(
            nuclei, elements, grid,
            max_iters=args.scf_iters, mixing=args.mixing, order=args.order,
        )
        print(f"Energy        : {point.energy:+.6f} Ha   (RHF + E_nn)")
        print(f"E_RHF         : {point.rhf_energy:+.6f} Ha")
        print(f"E_nn (pp)     : {point.nuclear_repulsion:+.6f} Ha")
        print(f"SCF iters     : {point.n_iters}   converged = {point.converged}")
        print(f"Orbital energy: {float(point.orbital_energies[0]):+.6f} Ha")
        return

    result = optimize_geometry(
        nuclei, elements, grid,
        geom_steps=args.geom_steps,
        learning_rate=args.lr,
        scf_max_iters=args.scf_iters,
        scf_mixing=args.mixing,
        order=args.order,
        verbose=True,
    )
    R_final = float(bond_length(result.final.nuclei, 0, 1))
    print()
    print(f"Final R_LiH   : {R_final:.4f}  a₀  ({R_final * 0.52917721:.3f} Å)")
    print(f"Final energy  : {result.final.energy:+.6f} Ha")


if __name__ == "__main__":
    main()
