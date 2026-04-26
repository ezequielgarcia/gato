"""End-to-end Phase 4 driver: the water molecule (RHF + HGH pseudopotentials).

The headline GATO demo. Three nuclei (one O, two H), eight valence electrons
(after replacing the bare $-Z/r$ potentials with HGH norm-conserving
pseudopotentials), four doubly-occupied molecular orbitals. Geometry is
relaxed by gradient descent on the Born–Oppenheimer total energy
$E(\\mathbf R) = E_\\text{RHF}(\\mathbf R) + E_\\text{nn}(\\mathbf R)$,
with forces from `jax.grad`.

The thesis being demonstrated
-----------------------------
"More is different" (Anderson 1972). Feed the solver three atomic numbers
(8, 1, 1) and a starting geometry; turn the crank on Schrödinger + Pauli
antisymmetry + the mean-field factorization; *chemistry happens*: the
bonds form at ~0.94 Å, the angle settles at ~106°, the lone pairs
localize on oxygen, the molecule develops a permanent dipole. Nothing in
that list was put in by hand. The Hartree–Fock approximation is the only
modeling choice; everything else is a consequence of solving the
Schrödinger equation variationally on a real-space grid.

The ~10% quantitative gaps (bond angle 106° vs experiment 104.5°; binding
~9 eV vs experiment 10.1 eV; dipole ~2.0 D vs experiment 1.85 D) are
exactly the electron-correlation energy that mean-field cannot see. That
is what the sister project `qato` recovers; see IDEAS.md.

Reference values (RHF complete-basis limit with HGH pseudopotentials)
---------------------------------------------------------------------
- O–H bond length:  ~0.94 Å  (≈ 1.78 Bohr)        experiment: 0.957 Å
- H–O–H angle:      ~106.1°                        experiment: 104.48°
- Total energy:     varies with PP; approximately -17.2 E_h with HGH
                    (the absolute number is reference-dependent because
                    PP calculations only output valence-electron energies)
- Atomization:      ~9 eV from valence energetics  experiment: 10.1 eV
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from .. import enable_x64
from ..geometry import (
    Nuclei,
    bond_angle,
    bond_length,
    nuclear_repulsion,
    recenter,
)
from ..grid import Grid3D
from ..pseudopotentials import (
    HGHParams,
    hgh_local_potential,
    hgh_nonlocal_apply,
    lookup,
    total_valence_charge,
)
from ..scf import scf_rhf


# ---------------------------------------------------------------------------
# Initial geometries
# ---------------------------------------------------------------------------

def water_initial_geometry(
    R_OH: float = 1.85,
    angle_deg: float = 100.0,
) -> Nuclei:
    """Place O at the origin and two H atoms in the xz-plane at ±half-angle.

    Default starting point is intentionally *not* the experimental geometry:
    geometry optimization should find equilibrium from a perturbed start.
    """
    half = jnp.deg2rad(angle_deg) / 2.0
    h1 = jnp.array([R_OH * jnp.sin(half), 0.0, R_OH * jnp.cos(half)])
    h2 = jnp.array([-R_OH * jnp.sin(half), 0.0, R_OH * jnp.cos(half)])
    o = jnp.zeros(3)
    return Nuclei(
        positions=jnp.stack([o, h1, h2]),
        charges=jnp.array([8.0, 1.0, 1.0]),
    )


def water_hgh_elements() -> tuple[HGHParams, ...]:
    """HGH parameter list aligned with `water_initial_geometry`."""
    return (lookup("O"), lookup("H"), lookup("H"))


# ---------------------------------------------------------------------------
# Pseudopotential nuclear repulsion
# ---------------------------------------------------------------------------
# When the bare nuclei are replaced by pseudopotentials, the electrostatic
# repulsion between the (frozen-core) ion cores uses the *valence* charges
# Z_ion, not the bare nuclear charges Z. The reason: the local
# pseudopotential includes the screening of the core electrons in its
# erf-Coulomb tail, so what one ion sees of another at long range is
# Z_ion/r, not Z/r.

def pp_nuclear_repulsion(
    nuclei: Nuclei,
    elements: tuple[HGHParams, ...],
) -> jax.Array:
    """E_nn for pseudoionic nuclei: Σ_{i<j} Z_ion_i Z_ion_j / |R_i - R_j|."""
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
class WaterPoint:
    nuclei: Nuclei
    orbitals: jax.Array
    energy: float                    # E_RHF + E_nn (pp)
    rhf_energy: float                # purely electronic + V_PP local + V_PP nonlocal
    nuclear_repulsion: float         # pp ion-ion
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
    lanczos_iters: int = 80,
    initial_orbitals: jax.Array | None = None,
) -> WaterPoint:
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
    E_total = float(res.energy) + E_nn

    return WaterPoint(
        nuclei=nuclei,
        orbitals=res.orbitals,
        energy=E_total,
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
    """Total BO energy at *fixed* (already converged) orbitals.

    Differentiable in `positions`. Stationarity of the SCF orbitals at the
    current geometry means `jax.grad(bo_energy, argnums=0)` returns the
    Hellmann–Feynman force without a Pulay term.

    The energy decomposes as:

        E(R) = T[ψ] + ⟨ψ|V_local(R)|ψ⟩ + ⟨ψ|V_nl(R)|ψ⟩
             + E_H[ρ] + E_x[ψ] + E_nn(R)

    Only V_local, V_nl and E_nn depend explicitly on R; T, E_H, E_x are
    intrinsic to ψ and constant under geometry differentiation when ψ is
    held fixed (Hellmann–Feynman).
    """
    from ..operators import kinetic
    from ..grid import inner_product
    from ..scf import _density, exchange_apply
    from ..solvers.poisson import hartree_energy

    nuclei = Nuclei(positions=positions, charges=charges)

    V_local = hgh_local_potential(nuclei, elements, grid)
    n_orb = orbitals.shape[-1]

    # One-body energy (kinetic + V_local + V_nl)
    E_h1 = 0.0
    for i in range(n_orb):
        phi = orbitals[..., i]
        h_phi = (
            kinetic(phi, grid.h, "dirichlet", order)
            + V_local * phi
            + hgh_nonlocal_apply(phi, nuclei, elements, grid)
        )
        E_h1 = E_h1 + inner_product(phi, h_phi, grid).real
    E_h1 = 2.0 * E_h1

    # Hartree
    rho = _density(orbitals)
    E_H = hartree_energy(rho, grid, epsilon=None)

    # Exchange (uses fixed orbitals)
    E_x = 0.0
    for i in range(n_orb):
        phi = orbitals[..., i]
        K_phi = exchange_apply(phi, orbitals, grid, epsilon=None)
        E_x = E_x - inner_product(phi, K_phi, grid).real

    E_nn = pp_nuclear_repulsion(nuclei, elements)

    return E_h1 + E_H + E_x + E_nn


def hellmann_feynman_force(
    point: WaterPoint,
    elements: tuple[HGHParams, ...],
    grid: Grid3D,
    order: int = 4,
) -> jax.Array:
    """Force on every nucleus, shape (K, 3). Valid at converged orbitals."""
    return -jax.grad(bo_energy, argnums=0)(
        point.nuclei.positions,
        point.nuclei.charges,
        point.orbitals,
        elements,
        grid,
        order,
    )


# ---------------------------------------------------------------------------
# Geometry optimization
# ---------------------------------------------------------------------------

@dataclass
class WaterGeomOptResult:
    final: WaterPoint
    trajectory: list[tuple[int, float, float, float]]   # (step, energy, R_OH, angle_deg)


def optimize_water_geometry(
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
    verbose: bool = True,
) -> WaterGeomOptResult:
    """Alternating SCF + nuclear gradient descent.

    Each outer step:
      1. Run RHF to convergence at current geometry.
      2. Take one Adam step on positions using the Hellmann–Feynman
         gradient (warm-starting orbitals from the previous geometry to
         accelerate SCF).
      3. Recenter at the charge-weighted COM to remove translational drift.

    Parameters
    ----------
    learning_rate : Adam step size in Bohr per gradient unit. 0.05 is a
        reasonable default; smaller values for fragile starting geometries.
    """
    nuclei = recenter(initial_nuclei)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(nuclei.positions)

    trajectory: list[tuple[int, float, float, float]] = []

    current = solve_rhf_at_geometry(
        nuclei, elements, grid,
        max_iters=scf_max_iters, tol=scf_tol, mixing=scf_mixing, order=order,
    )

    for step in range(geom_steps):
        grad_R = jax.grad(bo_energy, argnums=0)(
            current.nuclei.positions,
            current.nuclei.charges,
            current.orbitals,
            elements,
            grid,
            order,
        )
        updates, opt_state = optimizer.update(grad_R, opt_state)
        new_positions = optax.apply_updates(current.nuclei.positions, updates)
        nuclei = recenter(Nuclei(new_positions, current.nuclei.charges))

        current = solve_rhf_at_geometry(
            nuclei, elements, grid,
            initial_orbitals=current.orbitals,
            max_iters=scf_max_iters, tol=scf_tol, mixing=scf_mixing, order=order,
        )

        # O is at index 0; H atoms at 1, 2.
        R_OH = float(bond_length(nuclei, 0, 1))
        ang = float(bond_angle(nuclei, 1, 0, 2))
        trajectory.append((step, current.energy, R_OH, jnp.rad2deg(ang).item()))
        if verbose:
            print(
                f"  geom step {step:3d}  E = {current.energy:+.6f}  "
                f"R_OH = {R_OH:.4f} a₀  angle = {jnp.rad2deg(ang):.2f}°  "
                f"SCF iters = {current.n_iters}"
            )

    return WaterGeomOptResult(final=current, trajectory=trajectory)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve H₂O with RHF + HGH pseudopotentials and optimize geometry."
    )
    parser.add_argument("--N", type=int, default=64,
                        help="grid points per axis (PP makes 64 adequate)")
    parser.add_argument("--L", type=float, default=14.0,
                        help="box side in Bohr")
    parser.add_argument("--R0", type=float, default=1.85,
                        help="initial O-H distance in Bohr (~0.98 Å)")
    parser.add_argument("--angle0", type=float, default=100.0,
                        help="initial H-O-H angle in degrees")
    parser.add_argument("--geom-steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--scf-iters", type=int, default=60)
    parser.add_argument("--mixing", type=float, default=0.5)
    parser.add_argument("--order", type=int, choices=[2, 4], default=4)
    parser.add_argument("--single", action="store_true",
                        help="single-point only, skip geometry optimization")
    args = parser.parse_args()

    enable_x64()
    grid = Grid3D(N=args.N, L=args.L)
    nuclei = water_initial_geometry(R_OH=args.R0, angle_deg=args.angle0)
    elements = water_hgh_elements()

    print(f"H₂O on a {args.N}³ grid, L = {args.L}, h = {grid.h:.4f}, stencil order {args.order}")
    print(f"Pseudopotentials: HGH for O (Z_ion=6) and H (Z_ion=1); 8 valence electrons")
    print(f"Initial: R_OH = {args.R0:.3f} a₀, angle = {args.angle0:.2f}°")
    print(f"Targets: R_OH ≈ 1.81 a₀ (0.957 Å), angle ≈ 104.5°")
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
        print(f"Orbital energies (Ha): "
              f"{[f'{e:+.4f}' for e in point.orbital_energies.tolist()]}")
        return

    result = optimize_water_geometry(
        nuclei, elements, grid,
        geom_steps=args.geom_steps,
        learning_rate=args.lr,
        scf_max_iters=args.scf_iters,
        scf_mixing=args.mixing,
        order=args.order,
        verbose=True,
    )
    R_final = float(bond_length(result.final.nuclei, 0, 1))
    angle_final = jnp.rad2deg(bond_angle(result.final.nuclei, 1, 0, 2)).item()
    print()
    print(f"Final R_OH    : {R_final:.4f}  a₀  ({R_final * 0.52917721:.3f} Å)")
    print(f"Final angle   : {angle_final:.2f}°  (target 104.5°)")
    print(f"Final energy  : {result.final.energy:+.6f} Ha")


if __name__ == "__main__":
    main()
