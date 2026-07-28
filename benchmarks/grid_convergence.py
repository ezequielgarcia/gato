"""Grid-convergence study for the HGH pseudopotential molecules (H₂O, HCl).

Real-space grid results carry two independent discretization errors, and a
single (N, L) number cannot tell them apart:

  * **resolution error**, controlled by the spacing h = L/N. HGH projectors
    are narrow (r_ℓ ≈ 0.22 a₀ for O, 0.34–0.38 a₀ for Cl), so a grid that is
    comfortable for the valence density still under-resolves them.
  * **box error**, controlled by L. Dirichlet walls confine the orbitals and
    truncate the Hartree tail.

This benchmark separates them by sweeping one while holding the other fixed:

    energy    E(h)  at fixed L        → resolution error, extrapolated to h→0
    box       E(L)  at fixed h        → box error
    geometry  R(h), angle(h)          → does the *geometry* converge, not just E

Every sweep reports a fitted power law E(x) = E∞ + C·xᵖ so the extrapolated
value comes with an observed convergence order rather than an assumed one.

Run:
    uv run python -m benchmarks.grid_convergence --molecule water --mode energy
    uv run python -m benchmarks.grid_convergence --molecule hcl --mode box
    uv run python -m benchmarks.grid_convergence --molecule water --mode geometry

On a 12 GB card the practical ceiling is N ≈ 128 for a 4-orbital system: the
Lanczos Krylov buffer is (m+1)·N³ doubles and m itself grows with N (see
`gato.solvers.lanczos.default_krylov_dim`). Disabling the cuBLAS autotuner
(XLA_FLAGS=--xla_gpu_autotune_level=0) avoids a multi-GB profiling scratch
allocation at the largest sizes.
"""
from __future__ import annotations

import argparse
import time

import jax.numpy as jnp
import numpy as np

from gato import enable_x64
from gato.geometry import bond_angle, bond_length
from gato.grid import Grid3D
from gato.physics import hcl as hcl_mod
from gato.physics import water as water_mod


# ---------------------------------------------------------------------------
# Molecule adapters
# ---------------------------------------------------------------------------
# Each molecule exposes the same four hooks so the sweeps below stay generic.

class WaterCase:
    name = "H2O"
    default_L = 14.0
    # Near the relaxed geometry, so the reported energies are comparable to
    # published equilibrium numbers rather than to an arbitrary start point.
    R0 = 1.81
    angle0 = 104.5

    def nuclei(self):
        return water_mod.water_initial_geometry(R_OH=self.R0, angle_deg=self.angle0)

    def elements(self):
        return water_mod.water_hgh_elements()

    def solve(self, nuclei, grid, **kw):
        return water_mod.solve_rhf_at_geometry(nuclei, self.elements(), grid, **kw)

    def optimize(self, grid, **kw):
        res = water_mod.optimize_water_geometry(
            self.nuclei(), self.elements(), grid, verbose=False, **kw
        )
        n = res.final.nuclei
        return {
            "E": res.final.energy,
            "R": float(bond_length(n, 0, 1)),
            "angle": float(jnp.rad2deg(bond_angle(n, 1, 0, 2))),
            "converged": res.final.converged,
        }


class HClCase:
    name = "HCl"
    default_L = 14.0
    R0 = 2.41            # experimental R_e

    def nuclei(self):
        return hcl_mod.hcl_initial_geometry(R_HCl=self.R0)

    def elements(self):
        return hcl_mod.hcl_hgh_elements()

    def solve(self, nuclei, grid, **kw):
        return hcl_mod.solve_rhf_at_geometry(nuclei, self.elements(), grid, **kw)

    def optimize(self, grid, **kw):
        res = hcl_mod.optimize_geometry(
            self.nuclei(), self.elements(), grid, verbose=False, **kw
        )
        n = res.final.nuclei
        return {
            "E": res.final.energy,
            "R": float(bond_length(n, 0, 1)),
            "angle": float("nan"),
            "converged": res.final.converged,
        }


CASES = {"water": WaterCase, "hcl": HClCase}


# ---------------------------------------------------------------------------
# Power-law fit
# ---------------------------------------------------------------------------

def fit_power_law(xs, ys):
    """Fit y(x) = y∞ + C·xᵖ and return (y∞, C, p, rms_residual).

    p is found by a 1-D scan (the fit is linear in y∞ and C once p is fixed,
    so each candidate p costs one 2×2 least-squares solve). Scanning beats a
    generic optimizer here because there are only ever 4–7 data points and the
    residual surface in p is flat enough to trap gradient methods.

    Returns p = nan if fewer than three points are supplied.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")

    best = None
    for p in np.linspace(0.5, 6.0, 1101):
        A = np.column_stack([np.ones_like(xs), xs ** p])
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        resid = ys - A @ coef
        rms = float(np.sqrt(np.mean(resid ** 2)))
        if best is None or rms < best[3]:
            best = (float(coef[0]), float(coef[1]), float(p), rms)
    return best


def _report_fit(label, xs, ys, unit):
    y_inf, C, p, rms = fit_power_law(xs, ys)
    if np.isnan(p):
        print(f"  {label}: need >=3 points to fit")
        return None
    print(f"  {label}: extrapolated {y_inf:+.6f} {unit}   "
          f"(order p = {p:.2f}, rms resid = {rms:.2e})")
    return y_inf


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def sweep_energy(case, Ns, L, order, scf_iters):
    """E vs h at fixed L and fixed geometry."""
    print(f"{case.name}: energy vs grid spacing at fixed L = {L} a₀, "
          f"stencil order {order}")
    print(f"{'N':>5} {'h (a₀)':>9} {'E (Ha)':>15} {'SCF it':>7} {'conv':>6} {'time':>7}")
    rows = []
    nuclei = case.nuclei()
    for N in Ns:
        grid = Grid3D(N=N, L=L)
        t0 = time.perf_counter()
        pt = case.solve(nuclei, grid, max_iters=scf_iters, order=order)
        dt = time.perf_counter() - t0
        print(f"{N:5d} {grid.h:9.5f} {pt.energy:+15.8f} {pt.n_iters:7d} "
              f"{str(pt.converged):>6} {dt:6.0f}s", flush=True)
        if pt.converged:
            rows.append((grid.h, pt.energy))
        else:
            print(f"      ^ dropped from the fit: SCF did not converge")
    print()
    _report_fit("E(h -> 0)", [r[0] for r in rows], [r[1] for r in rows], "Ha")
    return rows


def sweep_box(case, Ls, target_h, order, scf_iters):
    """E vs L at (approximately) fixed h.

    N must be an integer, so h is held only as close to `target_h` as
    rounding allows; the achieved h is printed so any residual drift is
    visible rather than silently folded into the box-error estimate.

    That drift is not a rounding detail — it can swamp the measurement.
    Near h = 0.175 these molecules have dE/dh ≈ 1.6 Ha per unit h, so a 1.5%
    wobble in h shifts E by ~4 mHa, which is an order of magnitude larger than
    the box effect being looked for. **Choose target_h so that L/target_h is an
    integer for every L in the sweep** (e.g. h = 0.2 with L = 12, 14, 16, 18 →
    N = 60, 70, 80, 90). This routine warns when it cannot do so.
    """
    print(f"{case.name}: energy vs box size at fixed h ≈ {target_h} a₀, "
          f"stencil order {order}")
    print(f"{'L':>6} {'N':>5} {'h (a₀)':>9} {'E (Ha)':>15} {'SCF it':>7} "
          f"{'conv':>6} {'time':>7}")
    rows = []
    nuclei = case.nuclei()
    for L in Ls:
        N = int(round(L / target_h / 2)) * 2      # keep N even
        grid = Grid3D(N=N, L=L)
        t0 = time.perf_counter()
        pt = case.solve(nuclei, grid, max_iters=scf_iters, order=order)
        dt = time.perf_counter() - t0
        print(f"{L:6.1f} {N:5d} {grid.h:9.5f} {pt.energy:+15.8f} {pt.n_iters:7d} "
              f"{str(pt.converged):>6} {dt:6.0f}s", flush=True)
        if pt.converged:
            rows.append((L, pt.energy, grid.h))
    print()

    # If h drifted, say so loudly: the fit below cannot separate box error
    # from resolution error, and resolution almost always dominates.
    hs = [r[2] for r in rows]
    if hs and (max(hs) - min(hs)) > 1e-4:
        print(f"  WARNING: h drifted over {min(hs):.5f}..{max(hs):.5f} "
              f"({100 * (max(hs) - min(hs)) / min(hs):.1f}%). Box error and "
              f"resolution error are confounded; the fit below is unreliable.")
        print(f"  Re-run with a target_h that divides every L exactly.")
        # Points that happen to share an h *are* a controlled comparison.
        by_h: dict[float, list[tuple[float, float]]] = {}
        for L, E, h in rows:
            by_h.setdefault(round(h, 10), []).append((L, E))
        for h, group in sorted(by_h.items()):
            if len(group) >= 2:
                spread = max(e for _, e in group) - min(e for _, e in group)
                Ls = ", ".join(f"L={L:g}" for L, _ in group)
                print(f"  controlled at h = {h:.5f} ({Ls}): "
                      f"spread {spread * 1e3:.2f} mHa")

    # Box error decays with increasing L, so fit against 1/L.
    if len(rows) >= 3:
        _report_fit("E(L -> inf)", [1.0 / r[0] for r in rows],
                    [r[1] for r in rows], "Ha")
    return rows


def sweep_geometry(case, Ns, L, order, geom_steps, lr, scf_iters):
    """Relaxed geometry vs h — the observable the README actually claims."""
    print(f"{case.name}: relaxed geometry vs grid spacing at fixed L = {L} a₀")
    print(f"{'N':>5} {'h (a₀)':>9} {'R (a₀)':>10} {'R (Å)':>9} "
          f"{'angle':>9} {'E (Ha)':>15} {'conv':>6} {'time':>7}")
    rows = []
    for N in Ns:
        grid = Grid3D(N=N, L=L)
        t0 = time.perf_counter()
        out = case.optimize(
            grid, geom_steps=geom_steps, learning_rate=lr,
            scf_max_iters=scf_iters, order=order,
        )
        dt = time.perf_counter() - t0
        print(f"{N:5d} {grid.h:9.5f} {out['R']:10.5f} {out['R']*0.52917721:9.4f} "
              f"{out['angle']:9.3f} {out['E']:+15.8f} "
              f"{str(out['converged']):>6} {dt:6.0f}s", flush=True)
        # A geometry built on a non-converged final SCF is not a data point.
        if out["converged"]:
            rows.append((grid.h, out))
        else:
            print("      ^ dropped from the fit: final SCF did not converge")
    print()
    hs = [r[0] for r in rows]
    _report_fit("R(h -> 0)", hs, [r[1]["R"] for r in rows], "a₀")
    if not np.isnan(rows[0][1]["angle"]):
        _report_fit("angle(h -> 0)", hs, [r[1]["angle"] for r in rows], "deg")
    _report_fit("E(h -> 0)", hs, [r[1]["E"] for r in rows], "Ha")
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--molecule", choices=sorted(CASES), default="water")
    parser.add_argument("--mode", choices=["energy", "box", "geometry"],
                        default="energy")
    parser.add_argument("--N", type=int, nargs="+", default=None,
                        help="grid sizes to sweep (energy/geometry modes)")
    parser.add_argument("--L", type=float, default=None,
                        help="box side for energy/geometry modes")
    parser.add_argument("--box-L", type=float, nargs="+",
                        default=[12.0, 14.0, 16.0, 18.0],
                        help="box sides to sweep (box mode)")
    parser.add_argument("--target-h", type=float, default=0.175,
                        help="spacing held fixed in box mode")
    parser.add_argument("--order", type=int, choices=[2, 4], default=4)
    parser.add_argument("--scf-iters", type=int, default=80)
    parser.add_argument("--geom-steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()

    enable_x64()
    case = CASES[args.molecule]()
    L = args.L if args.L is not None else case.default_L

    if args.mode == "energy":
        Ns = args.N or [32, 48, 64, 80, 96, 112, 128]
        sweep_energy(case, Ns, L, args.order, args.scf_iters)
    elif args.mode == "box":
        sweep_box(case, args.box_L, args.target_h, args.order, args.scf_iters)
    else:
        Ns = args.N or [32, 48, 64, 80, 96]
        sweep_geometry(case, Ns, L, args.order, args.geom_steps, args.lr,
                       args.scf_iters)


if __name__ == "__main__":
    main()
