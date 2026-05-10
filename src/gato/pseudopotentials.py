"""HGH (Hartwigsen–Goedecker–Hutter) norm-conserving pseudopotentials.

A **norm-conserving pseudopotential** replaces the bare nuclear Coulomb
$-Z/r$ with a smooth, short-range potential that reproduces the correct
*valence* scattering of the all-electron atom while removing the deep
core orbitals from the calculation. For oxygen the 1s² core (which never
participates in chemistry) is frozen; the SCF only sees 6 valence
electrons instead of 8, and the singularity at the nucleus is replaced
by an analytic, smooth function. The required grid spacing relaxes from
$h \sim 0.05\,a_0$ (all-electron, to resolve the O 1s cusp) to
$h \sim 0.3$–$0.4\,a_0$ — a 6–8× linear, ~200–500× volumetric reduction
in grid points. This is the change that makes H₂O affordable on a
consumer GPU.

References
----------
- Goedecker, Teter, Hutter (GTH), Phys. Rev. B 54, 1703 (1996).
- Hartwigsen, Goedecker, Hutter (HGH), Phys. Rev. B 58, 3641 (1998).

Form
----
HGH pseudopotentials are **fully analytic** — every parameter is in the
paper, no interpolation tables, no external files. The total
pseudopotential is the sum of a *local* part and a *non-local separable*
part:

    V_PP(r, r') = V_loc(r) δ(r - r')  +  V_nl(r, r')

**Local part** (one expression per atom):

    V_loc(r) = -(Z_ion/r) erf(r / (√2 r_loc))
             + exp(-r²/(2 r_loc²)) · Σ_i C_i (r/r_loc)^{2(i-1)}

The erf-screened Coulomb cancels the bare $-Z_\text{ion}/r$ for $r \gg r_\text{loc}$
(correct long-range behaviour) and is finite at $r = 0$; the polynomial
correction tunes the short-range scattering. Pointwise multiplicative on
the grid — drops into the existing real-space SCF with no architectural
change.

**Non-local part** (separable Gaussian-radial projectors):

    V_nl |ψ⟩ = Σ_ℓ Σ_{i,j} |p_{ℓi}⟩ h_{ij}^{(ℓ)} ⟨p_{ℓj}|ψ⟩

with the analytic projector

    p_{ℓi}(r) = √2 · r^{ℓ + 2(i-1)} · exp(-r²/(2 r_ℓ²)) /
                ( r_ℓ^{ℓ + (4i-1)/2} · √Γ(ℓ + (4i-1)/2) )

Each ⟨p|ψ⟩ is one inner product on the grid; the action V_nl|ψ⟩ is a
small number of inner-products + outer-products per orbital per SCF
iteration. Differentiable in nuclear positions through the projector
construction, so Hellmann-Feynman forces flow naturally.

Parameter coverage
------------------
This module ships parameters for H, Li, O, and Cl. Adding a new element
is ~20 lines of tabulated values from the HGH paper. The parameter tables here use the LDA-fitted HGH values
(table I of Hartwigsen 1998); these are also commonly used with HF
without significant loss of accuracy because the pseudopotential's job
is to reproduce the valence-region radial logarithmic derivative, which
is dominated by kinetic + electrostatic effects rather than by the
exchange-correlation choice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.special import erf

from .geometry import Nuclei
from .grid import Grid3D, inner_product


# ---------------------------------------------------------------------------
# Parameter records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HGHProjector:
    """A single ℓ-channel of the non-local part.

    `r` is the projector's Gaussian width parameter $r_\\ell$. `h` is the
    symmetric matrix $h_{ij}^{(\\ell)}$ in Hartree. The number of radial
    projectors $n_\\ell$ in this channel is `h.shape[0]`.
    """

    ell: int
    r: float
    h: tuple[tuple[float, ...], ...]  # symmetric n×n matrix in Ha

    @property
    def n(self) -> int:
        return len(self.h)


@dataclass(frozen=True)
class HGHParams:
    """Full HGH parameter set for one element.

    `Z_ion` is the *valence* charge (= total Z minus frozen-core electrons).
    `c` are the local-part polynomial coefficients (C_1, C_2, C_3, C_4) in Ha.
    `projectors` is the per-ℓ non-local block; can be empty (e.g. for H).
    """

    symbol: str
    Z_ion: float
    r_loc: float
    c: tuple[float, float, float, float]
    projectors: tuple[HGHProjector, ...] = ()


# ---------------------------------------------------------------------------
# Tabulated parameters (Hartwigsen, Goedecker, Hutter 1998; LDA fit).
#
# H and O cover the H₂O target. Add elements here as needed; format is
# 1-to-1 with table I of the HGH paper.
# ---------------------------------------------------------------------------

HYDROGEN_HGH = HGHParams(
    symbol="H",
    Z_ion=1.0,
    r_loc=0.2,
    c=(-4.0663326, 0.6778322, 0.0, 0.0),
    projectors=(),  # no non-local part for H in the standard HGH tabulation
)


OXYGEN_HGH = HGHParams(
    symbol="O",
    Z_ion=6.0,
    r_loc=0.24762086,
    c=(-16.5803180, 2.39570952, 0.0, 0.0),
    projectors=(
        HGHProjector(ell=0, r=0.22178614, h=((18.26691718,),)),
        # The standard HGH O parameter set ships only an s-channel projector
        # with non-zero h; the p-channel block is identically zero so we omit it.
    ),
)


LITHIUM_HGH = HGHParams(
    symbol="Li",
    Z_ion=1.0,
    r_loc=0.78755305,
    c=(-1.89261247, 0.28605968, 0.0, 0.0),
    projectors=(
        HGHProjector(ell=0, r=0.66637518, h=((1.85881111,),)),
        HGHProjector(ell=1, r=1.07930561, h=((-0.00589504,),)),
    ),
)
# HGH-LDA "q1" Li (single 2s valence electron, 1s² core frozen). The s- and
# p-channel non-local projectors are essential — without them Li underbinds
# and the dimer geometry is off. Source: CP2K POTENTIAL data file
# (`Li GTH-PADE-q1`), traceable to Hartwigsen-Goedecker-Hutter 1998 table I.


CHLORINE_HGH = HGHParams(
    symbol="Cl",
    Z_ion=7.0,
    r_loc=0.41,
    c=(-6.86475431, 0.0, 0.0, 0.0),
    projectors=(
        # s-channel has TWO radial projectors with non-trivial h_12 coupling
        # — the first multi-i HGH block exercised in the codebase.
        HGHProjector(ell=0, r=0.33820832, h=(
            ( 9.06223968, -1.96193036),
            (-1.96193036,  5.06568240),
        )),
        HGHProjector(ell=1, r=0.37613709, h=((4.46587640,),)),
    ),
)
# HGH-LDA-q7 Cl (3s² 3p⁵ valence, [Ne] core frozen). Source: CP2K
# GTH_POTENTIALS file (`Cl GTH-PADE-q7`), traceable to Hartwigsen-Goedecker-
# Hutter 1998 table I.


HGH_TABLE: dict[str, HGHParams] = {
    "H": HYDROGEN_HGH,
    "Li": LITHIUM_HGH,
    "O": OXYGEN_HGH,
    "Cl": CHLORINE_HGH,
}


def lookup(symbol: str) -> HGHParams:
    if symbol not in HGH_TABLE:
        raise KeyError(
            f"No HGH parameters for {symbol!r}. Add them to gato.pseudopotentials "
            f"from Hartwigsen-Goedecker-Hutter 1998 table I."
        )
    return HGH_TABLE[symbol]


# ---------------------------------------------------------------------------
# Local part
# ---------------------------------------------------------------------------

def _hgh_local_one_center(
    grid_pts: jax.Array,           # (N, N, N, 3)
    R: jax.Array,                  # (3,)
    params: HGHParams,
) -> jax.Array:
    """Local pseudopotential V_loc(r) for a single nucleus at R."""
    diff = grid_pts - R
    r2 = jnp.sum(diff * diff, axis=-1)
    # Tiny floor to keep `erf(r/√2 r_loc)/r` well-defined at the origin
    # (limit r→0 of erf(r/a)/r is 2/(a √π)). Adding 1e-30 to r² is safe — at
    # r=0, erf(0)/0 returns the analytic limit through erf's Taylor expansion.
    r = jnp.sqrt(r2 + 1e-30)

    rho = r / params.r_loc
    # Erf-screened Coulomb: -Z_ion/r * erf(r/(√2 r_loc))
    # For r → 0 this is finite: -Z_ion · √(2/π) / r_loc.
    # For r → ∞ this approaches -Z_ion/r.
    sqrt2 = jnp.sqrt(2.0)
    erf_term = -params.Z_ion * erf(r / (sqrt2 * params.r_loc)) / r

    # Polynomial correction in (r/r_loc)²
    c1, c2, c3, c4 = params.c
    rho2 = rho * rho
    poly = c1 + c2 * rho2 + c3 * rho2 * rho2 + c4 * rho2 * rho2 * rho2
    poly_term = jnp.exp(-0.5 * rho2) * poly

    return erf_term + poly_term


def hgh_local_potential(
    nuclei: Nuclei,
    elements: tuple[HGHParams, ...],
    grid: Grid3D,
) -> jax.Array:
    """Sum of HGH local pseudopotentials over all nuclei.

    `elements[k]` is the HGHParams for nucleus k (must be aligned with
    `nuclei.positions[k]`). Returns a (N, N, N) array. Differentiable in
    `nuclei.positions`.
    """
    if len(elements) != int(nuclei.charges.shape[0]):
        raise ValueError(
            f"HGH parameter list length {len(elements)} != number of nuclei "
            f"{int(nuclei.charges.shape[0])}"
        )

    X, Y, Z = grid.coords()
    grid_pts = jnp.stack([X, Y, Z], axis=-1)

    V = jnp.zeros(grid.shape)
    for k, params in enumerate(elements):
        V = V + _hgh_local_one_center(grid_pts, nuclei.positions[k], params)
    return V


def total_valence_charge(elements: tuple[HGHParams, ...]) -> float:
    """Sum of valence charges; equals the number of explicitly treated electrons."""
    return float(sum(p.Z_ion for p in elements))


# ---------------------------------------------------------------------------
# Non-local part
# ---------------------------------------------------------------------------

def _projector_radial(r: jax.Array, ell: int, i_idx: int, r_ell: float) -> jax.Array:
    """Analytic HGH projector p_{ℓi}(r), returning a *radial* function value
    (not yet multiplied by an angular harmonic).

    p_{ℓi}(r) = √2 · r^{ℓ + 2(i-1)} · exp(-r²/(2 r_ℓ²))
                / (r_ℓ^{ℓ + (4i-1)/2} · √Γ(ℓ + (4i-1)/2))

    `i_idx` is 1-based to match the HGH paper.
    """
    n = ell + (4 * i_idx - 1) / 2.0
    norm = math.sqrt(2.0) / (r_ell ** n * math.sqrt(math.gamma(n)))
    power = ell + 2 * (i_idx - 1)
    return norm * (r ** power) * jnp.exp(-(r * r) / (2.0 * r_ell * r_ell))


def _angular_factor(ell: int, m: int, hat: jax.Array) -> jax.Array:
    """Real spherical harmonic Y_{ℓm}(r̂) up to ℓ=2.

    `hat` is the unit vector (..., 3). Returns a scalar field of the same
    leading shape as `hat`. Phase / normalization follows the standard
    Condon-Shortley real-spherical-harmonic convention.
    """
    if ell == 0 and m == 0:
        return jnp.full(hat.shape[:-1], 1.0 / math.sqrt(4.0 * math.pi))
    if ell == 1:
        c = math.sqrt(3.0 / (4.0 * math.pi))
        if m == -1:  # y
            return c * hat[..., 1]
        if m == 0:   # z
            return c * hat[..., 2]
        if m == 1:   # x
            return c * hat[..., 0]
    if ell == 2:
        x, y, z = hat[..., 0], hat[..., 1], hat[..., 2]
        if m == -2:  # xy
            return math.sqrt(15.0 / (4.0 * math.pi)) * x * y
        if m == -1:  # yz
            return math.sqrt(15.0 / (4.0 * math.pi)) * y * z
        if m == 0:   # 3z²-r²
            return 0.25 * math.sqrt(5.0 / math.pi) * (3.0 * z * z - 1.0)
        if m == 1:   # xz
            return math.sqrt(15.0 / (4.0 * math.pi)) * x * z
        if m == 2:   # x²-y²
            return 0.25 * math.sqrt(15.0 / math.pi) * (x * x - y * y)
    raise NotImplementedError(f"Real Y_{{{ell},{m}}} not implemented (need ℓ ≤ 2).")


def _projector_field(
    grid_pts: jax.Array,
    R: jax.Array,
    ell: int,
    m: int,
    i_idx: int,
    r_ell: float,
) -> jax.Array:
    """Full 3D projector: p_{ℓi}(r) Y_{ℓm}(r̂) centered on R. Shape (N, N, N)."""
    diff = grid_pts - R
    r2 = jnp.sum(diff * diff, axis=-1)
    r = jnp.sqrt(r2 + 1e-30)
    radial = _projector_radial(r, ell, i_idx, r_ell)
    hat = diff / r[..., None]
    angular = _angular_factor(ell, m, hat)
    return radial * angular


def _per_atom_projector_actions(
    psi: jax.Array,
    grid: Grid3D,
    grid_pts: jax.Array,
    R: jax.Array,
    params: HGHParams,
) -> jax.Array:
    """Sum over (ℓ, m, i, j) of |p_{ℓmi}⟩ h_ij^(ℓ) ⟨p_{ℓmj}|ψ⟩ for one nucleus."""
    out = jnp.zeros_like(psi)
    for proj in params.projectors:
        ell = proj.ell
        h = jnp.asarray(proj.h)
        n = proj.n
        for m in range(-ell, ell + 1):
            # Build the n radial projectors for this (ℓ, m) channel.
            ps = [
                _projector_field(grid_pts, R, ell, m, i + 1, proj.r)
                for i in range(n)
            ]
            # ⟨p_j | ψ⟩ for j = 1..n
            overlaps = jnp.stack([
                inner_product(p_j, psi, grid).real for p_j in ps
            ])  # (n,)
            # h @ overlaps
            coeffs = h @ overlaps  # (n,)
            for i in range(n):
                out = out + coeffs[i] * ps[i]
    return out


def hgh_nonlocal_apply(
    psi: jax.Array,
    nuclei: Nuclei,
    elements: tuple[HGHParams, ...],
    grid: Grid3D,
) -> jax.Array:
    """Action of the non-local pseudopotential on ψ:

        (V_nl ψ)(r) = Σ_k Σ_{ℓm} Σ_{ij} p_{ℓmi}^(k)(r) h_ij^(k,ℓ) ⟨p_{ℓmj}^(k)|ψ⟩.

    Differentiable in `nuclei.positions` and in `psi`. For atoms with no
    projectors (e.g. hydrogen) the contribution is identically zero.
    """
    X, Y, Z = grid.coords()
    grid_pts = jnp.stack([X, Y, Z], axis=-1)

    out = jnp.zeros_like(psi)
    for k, params in enumerate(elements):
        if not params.projectors:
            continue
        out = out + _per_atom_projector_actions(
            psi, grid, grid_pts, nuclei.positions[k], params,
        )
    return out


# ---------------------------------------------------------------------------
# Convenience: combined V_ext apply
# ---------------------------------------------------------------------------

def hgh_apply(
    psi: jax.Array,
    nuclei: Nuclei,
    elements: tuple[HGHParams, ...],
    grid: Grid3D,
    V_local: jax.Array | None = None,
) -> jax.Array:
    """Apply the full pseudopotential (local + non-local) to ψ.

    `V_local` may be precomputed and passed in to avoid recomputing the
    pointwise local field every Fock apply. If None it is built from
    `nuclei` and `elements`.
    """
    if V_local is None:
        V_local = hgh_local_potential(nuclei, elements, grid)
    return V_local * psi + hgh_nonlocal_apply(psi, nuclei, elements, grid)
