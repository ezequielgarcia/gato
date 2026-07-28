"""End-to-end Phase 4 driver: hydrogen chloride HCl (closed-shell RHF + HGH PP).

A polar heteronuclear diatomic that exercises a code path no earlier
phase reaches: the HGH non-local block of Cl has **two radial projectors
in the s-channel** with a non-trivial off-diagonal $h_{12}$ coupling.
H₂ uses no PP at all, LiH uses single-projector blocks, water's O block
is single-projector — HCl is the cheapest molecule that drives the
multi-i HGH machinery.

Pedagogical role
----------------
HCl is the textbook polar diatomic: a **mostly-covalent** bond with a
significant ionic component, sitting between the cleanly-covalent H₂ and
the near-ionic LiH. The dipole moment (experimental ~1.08 D) and the
asymmetric charge density on the Cl side are the visible chemistry. In
gas phase HCl is just a molecule; the *acid* behaviour HCl is famous for
is a solvent property and lives outside this driver — see IDEAS.md for
the natural Phase 5 follow-on (HCl·H₂O dimer, then implicit solvation).

Cost class
----------
8 valence electrons (H 1s¹ + Cl 3s² 3p⁵ valence after [Ne] core), so
$n_\\text{occ} = 4$ — same as water. Two atoms instead of three keeps
the geometry trivially 1D, but the SCF cost is comparable.

Reference values (RHF complete-basis with HGH-LDA pseudopotentials)
-------------------------------------------------------------------
- Bond length R_e:  ~2.41 a₀  (~1.275 Å)        experiment: 1.2745 Å
- Total energy:     ~-15.5 E_h with HGH         (PP-dependent absolute)
- Dipole moment:    ~1.5 D                       experiment: 1.08 D

Grid-refinement limit — READ BEFORE RAISING --N
-----------------------------------------------
**This driver is correct at its default N = 64 but cannot simply be refined.**
HCl is C∞v, so its occupied manifold contains an *exactly* 2-fold degenerate
3π level, and a single-vector Lanczos can only split a true degeneracy through
round-off. The Krylov dimension needed therefore far exceeds what
`default_krylov_dim` returns as the grid is refined:

    N = 64   needs ~120   (the default: OK)
    N = 80   needs ~200   (the default gives 120: WRONG ANSWER)

At N = 80 with the default the 3π pair does not appear at all, the fourth
orbital lands at -0.16 E_h instead of -0.48 E_h, and the total energy is off by
0.75 E_h — while `converged` still reports True. Refining N therefore makes the
answer *worse*, not better, until `lanczos_iters` is raised alongside it.

If you refine this molecule, pass `--lanczos-iters` explicitly (≳ 200 at
N = 80, more beyond) and check that the two highest occupied orbital energies
come out degenerate to ~1e-3 E_h. `scf_rhf` also warns when the final Ritz
residual shows the occupied states were not resolved.

For a grid-convergence study of a *heavier* atom without this pathology, a
C2v molecule such as H₂S is the right target: same n_occ = 4 and cost as water,
third-row diffuse valence, but no symmetry-required degeneracy. See
`benchmarks/grid_convergence.py` and `gato.solvers.lanczos.default_krylov_dim`.
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

def hcl_initial_geometry(R_HCl: float = 2.6) -> Nuclei:
    """H at -R/2, Cl at +R/2 along z. Default starts a touch longer than
    R_e (~2.41 a₀) so geometry-opt has a non-trivial trajectory.

    Note that `diatomic_along_z(R, Z1, Z2)` places the Z1 atom at -R/2
    and Z2 at +R/2; we pick Z1=1 (H) and Z2=17 (Cl) to keep the bare
    nuclear charges aligned with `hcl_hgh_elements()`.
    """
    return diatomic_along_z(R_HCl, Z1=1.0, Z2=17.0)


def hcl_hgh_elements() -> tuple[HGHParams, ...]:
    """HGH parameter list aligned with `hcl_initial_geometry`."""
    return (lookup("H"), lookup("Cl"))


# ---------------------------------------------------------------------------
# Pseudopotential nuclear repulsion
# ---------------------------------------------------------------------------

def pp_nuclear_repulsion(
    nuclei: Nuclei,
    elements: tuple[HGHParams, ...],
) -> jax.Array:
    """E_nn for pseudoionic nuclei: Σ_{i<j} Z_ion_i Z_ion_j / |R_i - R_j|.

    Uses the *valence* charges (Z_ion), not the bare nuclear charges,
    since each ion's long-range tail is screened by its frozen core.
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
class HClPoint:
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
    max_iters: int = 60,
    tol: float = 1e-6,
    mixing: float = 0.5,
    order: int = 4,
    lanczos_iters: int | None = None,
    initial_orbitals: jax.Array | None = None,
) -> HClPoint:
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
    return HClPoint(
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
class HClGeomOptResult:
    final: HClPoint
    trajectory: list[tuple[int, float, float]]   # (step, energy, R_HCl)


def optimize_geometry(
    initial_nuclei: Nuclei,
    elements: tuple[HGHParams, ...],
    grid: Grid3D,
    *,
    geom_steps: int = 30,
    learning_rate: float = 0.05,
    scf_max_iters: int = 60,
    scf_tol: float = 1e-6,
    scf_mixing: float = 0.5,
    order: int = 4,
    lanczos_iters: int | None = None,
    verbose: bool = True,
) -> HClGeomOptResult:
    """Alternating SCF + nuclear gradient descent (Adam on positions)."""
    def solve_at(nuclei: Nuclei, prev: HClPoint | None) -> HClPoint:
        init_orb = prev.orbitals if prev is not None else None
        return solve_rhf_at_geometry(
            nuclei, elements, grid,
            initial_orbitals=init_orb,
            max_iters=scf_max_iters, tol=scf_tol, mixing=scf_mixing, order=order,
            lanczos_iters=lanczos_iters,
        )

    def bo_grad(p: HClPoint) -> jax.Array:
        return _bo_grad_positions(
            p.nuclei.positions, p.nuclei.charges, p.orbitals,
            elements, grid, order,
        )

    trajectory: list[tuple[int, float, float]] = []

    def on_step(step: int, p: HClPoint) -> None:
        R = float(bond_length(p.nuclei, 0, 1))
        trajectory.append((step, p.energy, R))
        if verbose:
            print(
                f"  geom step {step:3d}  E = {p.energy:+.6f}  "
                f"R_HCl = {R:.4f} a₀  SCF iters = {p.n_iters}"
            )

    final, _ = optimize_geometry_alternating(
        initial_nuclei,
        solve_at_geometry=solve_at,
        bo_grad=bo_grad,
        geom_steps=geom_steps,
        learning_rate=learning_rate,
        on_step=on_step,
    )
    return HClGeomOptResult(final=final, trajectory=trajectory)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve HCl with RHF + HGH pseudopotentials and optimize geometry."
    )
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--L", type=float, default=14.0)
    parser.add_argument("--R0", type=float, default=2.6,
                        help="initial H-Cl distance in Bohr (R_e ≈ 2.41)")
    parser.add_argument("--geom-steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--scf-iters", type=int, default=60)
    parser.add_argument("--mixing", type=float, default=0.5)
    parser.add_argument("--order", type=int, choices=[2, 4], default=4)
    parser.add_argument("--lanczos-iters", type=int, default=None,
                        help="Krylov dimension (default: scales with --N)")
    parser.add_argument("--single", action="store_true",
                        help="single-point only, skip geometry optimization")
    args = parser.parse_args()

    enable_x64()
    grid = Grid3D(N=args.N, L=args.L)
    nuclei = hcl_initial_geometry(R_HCl=args.R0)
    elements = hcl_hgh_elements()

    print(f"HCl on a {args.N}³ grid, L = {args.L}, h = {grid.h:.4f}, stencil order {args.order}")
    print(f"Pseudopotentials: HGH H (Z_ion=1) + HGH-LDA-q7 Cl (Z_ion=7); 8 valence electrons")
    print(f"Initial: R_HCl = {args.R0:.3f} a₀     target R_e ≈ 2.41 a₀ (1.275 Å)")
    print()

    if args.single:
        point = solve_rhf_at_geometry(
            nuclei, elements, grid,
            max_iters=args.scf_iters, mixing=args.mixing, order=args.order,
            lanczos_iters=args.lanczos_iters,
        )
        print(f"Energy        : {point.energy:+.6f} Ha   (RHF + E_nn)")
        print(f"E_RHF         : {point.rhf_energy:+.6f} Ha")
        print(f"E_nn (pp)     : {point.nuclear_repulsion:+.6f} Ha")
        print(f"SCF iters     : {point.n_iters}   converged = {point.converged}")
        print("Orbital energies (Ha):")
        for i, e in enumerate(point.orbital_energies):
            print(f"  φ_{i}: {float(e):+.6f}")
        return

    result = optimize_geometry(
        nuclei, elements, grid,
        geom_steps=args.geom_steps,
        learning_rate=args.lr,
        scf_max_iters=args.scf_iters,
        scf_mixing=args.mixing,
        order=args.order,
        lanczos_iters=args.lanczos_iters,
        verbose=True,
    )
    R_final = float(bond_length(result.final.nuclei, 0, 1))
    print()
    print(f"Final R_HCl   : {R_final:.4f}  a₀  ({R_final * 0.52917721:.3f} Å)")
    print(f"Final energy  : {result.final.energy:+.6f} Ha")


if __name__ == "__main__":
    main()
