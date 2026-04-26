"""Restricted Hartree–Fock self-consistent-field solver on a real-space grid.

This is the **single SCF method** in GATO. The project is deliberately scoped
to ab-initio mean-field RHF — no DFT, no XC functionals — so that every
emergent observable (geometry, polarity, hybridization, atomization energy)
can be traced to exactly three ingredients: Schrödinger, Pauli antisymmetry
(Slater determinant), and the mean-field factorization. Correlation lives in
the sibling project `qato` (variational Monte Carlo on He–He); see IDEAS.md.


Closed-shell RHF: each spatial orbital $\\phi_i$ is doubly occupied (one spin-up,
one spin-down electron). The total density is

    ρ(r) = 2 Σ_i |φ_i(r)|²,

and the Fock operator acting on any spatial orbital is

    F φ_i = ĥ φ_i + J[ρ] φ_i − K̂ φ_i,

with one-electron operator $\\hat h = -\\tfrac{1}{2}\\nabla^2 + V_\\text{ext}$,
Hartree potential $J[\\rho](r) = \\int \\rho(r')/|r-r'|\\,dV'$, and the exchange
operator

    (K̂ ψ)(r) = Σ_j φ_j(r) · J[φ_j^* ψ](r).

The exchange operator is written in general $n_\\text{orb}^2$ form (one Poisson
solve per (j, ψ) pair), so the same code is used unchanged in Phase 4 for
molecules with multiple occupied orbitals.

Total RHF energy (for orthonormal orbitals):

    E = 2 Σ_i ⟨φ_i|ĥ|φ_i⟩ + E_H[ρ] + E_x,

with Hartree self-energy $E_H[ρ] = \\tfrac{1}{2}\\int \\rho J[\\rho]\\,dV$ and
exchange energy $E_x = -\\sum_i ⟨φ_i | K̂ φ_i⟩$. The identity
$E_x = -\\sum_{ij} ⟨ij|ji⟩$ is recovered automatically because
$⟨φ_i|K̂ φ_i⟩ = \\sum_j ⟨ij|ji⟩$.

SCF loop: diagonalize the Fock built from the previous iteration's orbitals
(matrix-free Lanczos), take the lowest $n_\\text{occ}$ eigenstates as the new
orbitals, and repeat until the energy is stationary. Density-based linear
mixing damps oscillations when `mixing < 1`; the default of 1.0 (no mixing) is
sufficient for atoms like helium with a well-conditioned initial guess.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from .grid import Grid3D, inner_product, integrate
from .hamiltonian import Hamiltonian
from .operators import kinetic
from .solvers.lanczos import lanczos
from .solvers.poisson import hartree_energy, hartree_potential


def _density(orbitals: jax.Array) -> jax.Array:
    """ρ(r) = 2 Σ_i |φ_i(r)|² for closed-shell occupation."""
    return 2.0 * jnp.sum(jnp.abs(orbitals) ** 2, axis=-1)


def exchange_apply(
    psi: jax.Array,
    orbitals: jax.Array,
    grid: Grid3D,
    epsilon: float | None = None,
) -> jax.Array:
    """General exchange action (K̂ ψ)(r) = Σ_j φ_j(r) · J[φ_j* ψ](r).

    Parameters
    ----------
    psi : (N, N, N) input wavefunction.
    orbitals : (N, N, N, n_orb) occupied spatial orbitals.
    grid : Grid3D on which both live.
    epsilon : softening scale for the Hartree kernel. Defaults to h/2.
    """
    def one_term(phi_j: jax.Array) -> jax.Array:
        return phi_j * hartree_potential(jnp.conj(phi_j) * psi, grid, epsilon=epsilon)

    contribs = jax.vmap(one_term, in_axes=-1, out_axes=-1)(orbitals)
    return jnp.sum(contribs, axis=-1)


@dataclass(frozen=True)
class RHFFock:
    """Closed-shell Fock operator as a matrix-free linear map.

    Duck-types as a `Hamiltonian` (exposes `.grid` and `.apply`), so the
    existing Lanczos solver accepts it without modification.

    `V_ext` is the multiplicative external (or pseudopotential local) part.
    `V_nl_apply`, when provided, is a callable ψ → V_nl ψ implementing the
    non-local pseudopotential action; pass `None` for the all-electron case.
    """

    grid: Grid3D
    V_ext: jax.Array
    orbitals: jax.Array
    boundary: str = "dirichlet"
    order: int = 4
    epsilon: float | None = None
    V_nl_apply: Callable[[jax.Array], jax.Array] | None = None

    def density(self) -> jax.Array:
        return _density(self.orbitals)

    def hartree(self) -> jax.Array:
        return hartree_potential(self.density(), self.grid, epsilon=self.epsilon)

    def apply(self, psi: jax.Array) -> jax.Array:
        h_psi = kinetic(psi, self.grid.h, self.boundary, self.order) + self.V_ext * psi
        if self.V_nl_apply is not None:
            h_psi = h_psi + self.V_nl_apply(psi)
        J_psi = self.hartree() * psi
        K_psi = exchange_apply(psi, self.orbitals, self.grid, epsilon=self.epsilon)
        return h_psi + J_psi - K_psi


def rhf_energy(
    orbitals: jax.Array,
    V_ext: jax.Array,
    grid: Grid3D,
    boundary: str = "dirichlet",
    order: int = 4,
    epsilon: float | None = None,
    V_nl_apply: Callable[[jax.Array], jax.Array] | None = None,
) -> jax.Array:
    """Total closed-shell RHF energy E = 2 Σ_i ⟨φ_i|ĥ|φ_i⟩ + E_H[ρ] + E_x.

    `epsilon` controls the softening of the Hartree/exchange kernel; it is
    independent of the softening baked into `V_ext` so that both can be swept
    together when extrapolating to the bare-Coulomb limit.

    `V_nl_apply`, if provided, contributes 2 Σ_i ⟨φ_i|V_nl|φ_i⟩ to the
    one-body energy. Used for non-local pseudopotentials.
    """
    n_orb = orbitals.shape[-1]

    E_h1 = 0.0
    for i in range(n_orb):
        phi = orbitals[..., i]
        h_phi = kinetic(phi, grid.h, boundary, order) + V_ext * phi
        if V_nl_apply is not None:
            h_phi = h_phi + V_nl_apply(phi)
        E_h1 = E_h1 + inner_product(phi, h_phi, grid).real
    E_h1 = 2.0 * E_h1

    E_H = hartree_energy(_density(orbitals), grid, epsilon=epsilon)

    E_x = 0.0
    for i in range(n_orb):
        phi = orbitals[..., i]
        K_phi = exchange_apply(phi, orbitals, grid, epsilon=epsilon)
        E_x = E_x - inner_product(phi, K_phi, grid).real

    return E_h1 + E_H + E_x


def _default_initial_orbitals(
    V_ext: jax.Array,
    grid: Grid3D,
    n_occ: int,
    boundary: str,
    order: int,
    lanczos_iters: int,
) -> jax.Array:
    """Core-Hamiltonian guess: diagonalize ĥ = T + V_ext and take its n_occ
    lowest states.

    The Lanczos Krylov subspace is generated from a single starter vector, so
    the starter must have nonzero overlap with every eigenstate we want to
    find. For a purely spherical starter like $e^{-r}$ the subspace contains
    only $\\ell = 0$ functions and p-orbitals are invisible. To support
    atoms and molecules with p (and eventually d) occupancy without bespoke
    symmetry work, we use $e^{-r}$ plus a small asymmetric perturbation
    that breaks cubic symmetry — enough to seed every irreducible
    representation of the Hamiltonian's point group.
    """
    H_core = Hamiltonian(grid=grid, V=V_ext, boundary=boundary, order=order)
    X, Y, Z = grid.coords()
    r = jnp.sqrt(X * X + Y * Y + Z * Z + 1e-6)
    # Core (tight) + valence (diffuse) envelopes → overlap with both deep and
    # shallow states when Z is large and the core orbital is short-ranged.
    core = jnp.exp(-4.0 * r)
    valence = jnp.exp(-r)
    # Symmetry-breaking perturbation: (x + y + z) seeds p; (x·y + y·z + x·z)
    # seeds d; small amplitude keeps the starter dominated by the symmetric
    # envelope so convergence of the low-lying s-states is unharmed.
    perturbation = 0.1 * (X + Y + Z) + 0.02 * (X * Y + Y * Z + X * Z)
    psi0 = (core + valence) * (1.0 + perturbation)
    res = lanczos(H_core, psi0, n_iters=lanczos_iters, n_eigenstates=n_occ)
    return res.eigenstates


@dataclass
class SCFResult:
    orbitals: jax.Array            # (N, N, N, n_occ)
    energy: float                  # total RHF energy
    orbital_energies: jax.Array    # (n_occ,) — Fock eigenvalues at convergence
    n_iters: int
    converged: bool
    energy_history: list[float]


def scf_rhf(
    V_ext: jax.Array,
    grid: Grid3D,
    n_occ: int,
    *,
    initial_orbitals: jax.Array | None = None,
    max_iters: int = 50,
    tol: float = 1e-6,
    mixing: float = 1.0,
    boundary: str = "dirichlet",
    order: int = 4,
    lanczos_iters: int = 60,
    epsilon: float | None = None,
    V_nl_apply: Callable[[jax.Array], jax.Array] | None = None,
) -> SCFResult:
    """Drive the closed-shell RHF SCF loop to self-consistency.

    Parameters
    ----------
    V_ext : one-electron external potential on the grid, shape (N, N, N).
        For all-electron calculations this is the (softened) Coulomb sum;
        for pseudopotential calculations this is the local part of the
        pseudopotential.
    grid : Grid3D the potential is sampled on.
    n_occ : number of doubly-occupied spatial orbitals (= electrons / 2).
    initial_orbitals : optional (N, N, N, n_occ) guess. Defaults to the
        core-Hamiltonian eigenstates, which is exact for one electron and a
        reasonable starting point otherwise.
    max_iters : hard ceiling on SCF iterations.
    tol : convergence threshold on |E_k − E_{k−1}| (Hartree).
    mixing : linear density mix, α ∈ (0, 1]; the Fock is built from
        ρ_k = α ρ_new + (1−α) ρ_old. α=1 (default) is no mixing. Reducing α
        damps oscillations for hard-to-converge systems.
    boundary, order : passed through to the kinetic stencil.
    lanczos_iters : Krylov dimension for each Fock diagonalization.
    V_nl_apply : optional non-local pseudopotential action ψ → V_nl ψ.
    """
    if initial_orbitals is None:
        orbitals = _default_initial_orbitals(
            V_ext, grid, n_occ, boundary, order, lanczos_iters
        )
    else:
        orbitals = initial_orbitals

    history: list[float] = []
    rho = _density(orbitals)

    E_prev = float("inf")
    E = float("inf")
    orbital_energies = jnp.zeros(n_occ)
    converged = False
    it = 0

    for it in range(1, max_iters + 1):
        # Build the Fock with the (possibly mixed) density but current orbitals
        # for the exchange piece. For linear mixing we replace the density that
        # enters the Hartree term; for the exchange operator we use the current
        # orbitals directly, which is the standard Pulay / density-mixing choice.
        V_H = hartree_potential(rho, grid, epsilon=epsilon)
        fock = _MixedFock(
            grid=grid,
            V_ext=V_ext,
            orbitals=orbitals,
            V_H=V_H,
            boundary=boundary,
            order=order,
            epsilon=epsilon,
            V_nl_apply=V_nl_apply,
        )
        res = lanczos(fock, orbitals[..., 0], n_iters=lanczos_iters, n_eigenstates=n_occ)
        new_orbitals = res.eigenstates
        orbital_energies = res.eigenvalues

        rho_new = _density(new_orbitals)
        rho = mixing * rho_new + (1.0 - mixing) * rho
        orbitals = new_orbitals

        E = float(rhf_energy(
            orbitals, V_ext, grid, boundary, order,
            epsilon=epsilon, V_nl_apply=V_nl_apply,
        ))
        history.append(E)

        if abs(E - E_prev) < tol:
            converged = True
            break
        E_prev = E

    return SCFResult(
        orbitals=orbitals,
        energy=E,
        orbital_energies=orbital_energies,
        n_iters=it,
        converged=converged,
        energy_history=history,
    )


@dataclass(frozen=True)
class _MixedFock:
    """Fock operator with a precomputed Hartree potential (from a possibly-mixed
    density) and exchange built on the current orbitals.

    The Hartree potential array `V_H` is stored, not the density — so every
    `apply` call is one kinetic + two multiplies + the n_orb Poisson solves
    that the exchange operator requires, rather than also redoing `J[ρ]`.

    `V_nl_apply` carries the non-local pseudopotential action when present.
    """

    grid: Grid3D
    V_ext: jax.Array
    orbitals: jax.Array
    V_H: jax.Array
    boundary: str = "dirichlet"
    order: int = 4
    epsilon: float | None = None
    V_nl_apply: Callable[[jax.Array], jax.Array] | None = None

    def apply(self, psi: jax.Array) -> jax.Array:
        h_psi = kinetic(psi, self.grid.h, self.boundary, self.order) + self.V_ext * psi
        if self.V_nl_apply is not None:
            h_psi = h_psi + self.V_nl_apply(psi)
        J_psi = self.V_H * psi
        K_psi = exchange_apply(psi, self.orbitals, self.grid, epsilon=self.epsilon)
        return h_psi + J_psi - K_psi
