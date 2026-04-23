"""Exchange–correlation functionals for Kohn–Sham DFT (closed-shell / unpolarized).

Currently supported:

- **LDA exchange** (Dirac): $\\epsilon_x^{LDA}(\\rho) = -C_x\\,\\rho^{1/3}$ with
  $C_x = \\tfrac{3}{4}(3/\\pi)^{1/3}$.  Energy $E_x = -C_x\\int\\rho^{4/3}\\,dV$,
  potential $V_x = -\\tfrac{4}{3}C_x\\,\\rho^{1/3}$.

- **LDA correlation** (Perdew–Zunger 1981): a piecewise-rational fit to the
  Ceperley–Alder 1980 QMC data on the homogeneous electron gas.
  Implemented for the unpolarized / spin-restricted case — all that Phase 3–5
  need — with the high-density ($r_s < 1$) RPA-like logarithmic form and the
  low-density ($r_s \\geq 1$) rational form, smoothly matched at $r_s = 1$.

Each functional is exposed in two forms:

- ``energy(rho, grid)``: the extensive energy $E_\\ast[\\rho] = \\int \\rho \\,
  \\epsilon_\\ast(\\rho)\\,dV$.
- ``potential(rho)``: the pointwise array $V_\\ast(r) = \\delta E_\\ast / \\delta \\rho$.

The combined ``exchange_correlation_*`` wrappers sum exchange + correlation so
callers (the KS-DFT SCF loop) need only one call each.

References
----------
- Dirac, P. A. M. (1930), Proc. Cambridge Phil. Soc. **26**, 376.
- Perdew, J. P. & Zunger, A. (1981), Phys. Rev. B **23**, 5048.
- Ceperley, D. M. & Alder, B. J. (1980), Phys. Rev. Lett. **45**, 566.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .grid import Grid3D, integrate

# Dirac exchange constant C_x = (3/4) (3/π)^(1/3)
_CX = 0.75 * (3.0 / jnp.pi) ** (1.0 / 3.0)

# Perdew–Zunger 1981 unpolarized parameters
_PZ_A, _PZ_B, _PZ_C, _PZ_D = 0.0311, -0.0480, 0.0020, -0.0116
_PZ_GAMMA, _PZ_BETA1, _PZ_BETA2 = -0.1423, 1.0529, 0.3334

# Numerical floor to keep ρ^(1/3) and log ρ well-defined on the grid. Densities
# below this threshold contribute negligibly to E_xc but would produce NaNs in
# the logarithm of the PZ high-density branch or in r_s = ∞ limit.
_RHO_FLOOR = 1e-20


def _rs(rho: jax.Array) -> jax.Array:
    """Wigner–Seitz radius r_s = (3 / (4π ρ))^(1/3) at each grid point."""
    return (3.0 / (4.0 * jnp.pi * jnp.maximum(rho, _RHO_FLOOR))) ** (1.0 / 3.0)


# ---- LDA exchange (Dirac) ------------------------------------------------- #

def lda_exchange_energy_density(rho: jax.Array) -> jax.Array:
    """$\\epsilon_x(\\rho) = -C_x \\rho^{1/3}$ — the per-electron exchange energy."""
    return -_CX * jnp.maximum(rho, _RHO_FLOOR) ** (1.0 / 3.0)


def lda_exchange_potential(rho: jax.Array) -> jax.Array:
    """$V_x(r) = -\\tfrac{4}{3} C_x \\rho(r)^{1/3}$."""
    return -(4.0 / 3.0) * _CX * jnp.maximum(rho, _RHO_FLOOR) ** (1.0 / 3.0)


def lda_exchange_energy(rho: jax.Array, grid: Grid3D) -> jax.Array:
    """$E_x^{LDA}[\\rho] = \\int \\rho\\,\\epsilon_x(\\rho)\\,dV = -C_x \\int \\rho^{4/3}\\,dV$."""
    return integrate(rho * lda_exchange_energy_density(rho), grid)


# ---- LDA correlation (Perdew–Zunger 1981) --------------------------------- #

def _eps_c_pz(rs: jax.Array) -> jax.Array:
    """Per-electron correlation energy ε_c(r_s), unpolarized PZ81 form."""
    high = _PZ_A * jnp.log(rs) + _PZ_B + _PZ_C * rs * jnp.log(rs) + _PZ_D * rs
    low = _PZ_GAMMA / (1.0 + _PZ_BETA1 * jnp.sqrt(rs) + _PZ_BETA2 * rs)
    return jnp.where(rs < 1.0, high, low)


def _dEps_c_drs(rs: jax.Array) -> jax.Array:
    """dε_c/dr_s — needed for V_c via V_c = ε_c − (r_s/3) dε_c/dr_s."""
    high = _PZ_A / rs + _PZ_C * jnp.log(rs) + _PZ_C + _PZ_D
    sqrt_rs = jnp.sqrt(rs)
    denom = 1.0 + _PZ_BETA1 * sqrt_rs + _PZ_BETA2 * rs
    low = -_PZ_GAMMA * (_PZ_BETA1 / (2.0 * sqrt_rs) + _PZ_BETA2) / (denom * denom)
    return jnp.where(rs < 1.0, high, low)


def lda_correlation_energy_density(rho: jax.Array) -> jax.Array:
    """ε_c(ρ) — PZ81 correlation energy per electron."""
    return _eps_c_pz(_rs(rho))


def lda_correlation_potential(rho: jax.Array) -> jax.Array:
    """$V_c(r) = \\epsilon_c - \\tfrac{r_s}{3}\\,d\\epsilon_c/dr_s$.

    Follows from $V_c = d(\\rho \\epsilon_c)/d\\rho$ and $d/d\\rho = -(r_s/3\\rho)\\,d/dr_s$.
    """
    rs = _rs(rho)
    return _eps_c_pz(rs) - (rs / 3.0) * _dEps_c_drs(rs)


def lda_correlation_energy(rho: jax.Array, grid: Grid3D) -> jax.Array:
    return integrate(rho * lda_correlation_energy_density(rho), grid)


# ---- Combined LDA XC wrappers --------------------------------------------- #

def lda_xc_potential(rho: jax.Array) -> jax.Array:
    """V_xc(r) = V_x(r) + V_c(r), the full LDA Kohn–Sham XC potential."""
    return lda_exchange_potential(rho) + lda_correlation_potential(rho)


def lda_xc_energy(rho: jax.Array, grid: Grid3D) -> jax.Array:
    """E_xc[ρ] = E_x[ρ] + E_c[ρ]."""
    return lda_exchange_energy(rho, grid) + lda_correlation_energy(rho, grid)
