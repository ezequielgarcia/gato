"""Extrapolate helium's RHF energy to zero softening.

On a cell-centered 3D grid the softened Coulomb potential V_ε(r) = -Z/√(r² + ε²)
and the softened Hartree/exchange kernel 1/√(r² + ε²) both carry a positive
energy bias that grows with Z. For helium at ε = h/2 on an N=64, L=10 grid this
bias is ≈ 260 mHa; the SCF solves the softened Hamiltonian to 10⁻⁶ Hartree but
the softened Hamiltonian itself is not the true Coulomb Hamiltonian.

Sweeping ε at fixed (N, L) and fitting

    E(ε) = E₀ + a·ε + b·ε² + O(ε³)

extrapolates the converged SCF energy to the ε → 0 limit. The leading
coefficient is O(ε) (logarithmic integrand from the 1/r singularity), not O(ε²),
so `linear_quadratic` is the default fit model.

Output:
    - stdout table of E(ε) for RHF
    - extrapolated E₀ under linear, quadratic, and linear_quadratic fits
    - residual vs. published reference RHF (-2.862 E_h)
    - optional plot at docs/figures/helium_softening_extrapolation.png

Run:
    uv run python -m benchmarks.helium_softening_extrapolation
    uv run python -m benchmarks.helium_softening_extrapolation --plot
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import gato
from gato.grid import Grid3D
from gato.potentials import softened_coulomb
from gato.scf import scf_rhf


# Published references (closed-shell helium ground state, non-relativistic Coulomb)
_E_RHF_REF = -2.8617
_E_EXACT = -2.9037


def run_sweep(
    N: int,
    L: float,
    epsilons: list[float],
    mixing: float,
    max_iters: int,
    order: int,
    lanczos_iters: int,
) -> list[tuple[float, float, int, bool]]:
    """Run one RHF SCF per ε, at fixed (N, L). Returns (ε, E, n_iter, converged)."""
    grid = Grid3D(N=N, L=L)
    results = []
    for eps in epsilons:
        V_ext = softened_coulomb(grid, Z=2.0, epsilon=eps)
        res = scf_rhf(
            V_ext,
            grid,
            n_occ=1,
            max_iters=max_iters,
            tol=1e-6,
            mixing=mixing,
            order=order,
            lanczos_iters=lanczos_iters,
            epsilon=eps,
        )
        results.append((float(eps), float(res.energy), int(res.n_iters), bool(res.converged)))
        status = "✓" if res.converged else "✗"
        print(
            f"  ε = {eps:.5f}   E = {res.energy:+.5f} Ha   "
            f"iters={res.n_iters:2d} {status}"
        )
    return results


def extrapolate_to_zero(
    results: list[tuple[float, float, int, bool]],
    model: str = "linear_quadratic",
) -> tuple[float, np.ndarray]:
    """Fit E(ε) and return (E₀, coefficients)."""
    eps = np.array([r[0] for r in results])
    E = np.array([r[1] for r in results])
    if model == "quadratic":
        A = np.column_stack([np.ones_like(eps), eps**2])
    elif model == "linear":
        A = np.column_stack([np.ones_like(eps), eps])
    elif model == "linear_quadratic":
        A = np.column_stack([np.ones_like(eps), eps, eps**2])
    else:
        raise ValueError(f"unknown model {model!r}")
    coef, *_ = np.linalg.lstsq(A, E, rcond=None)
    return float(coef[0]), coef


def maybe_plot(
    rhf_results,
    rhf_extrap,
    out_path: Path,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    eps_rhf = np.array([r[0] for r in rhf_results])
    E_rhf = np.array([r[1] for r in rhf_results])
    ax.plot(eps_rhf, E_rhf, "o-", label="RHF (measured)", color="C0")
    ax.axhline(rhf_extrap, color="C0", linestyle="--", alpha=0.6, label=f"RHF ε→0 fit = {rhf_extrap:+.4f}")
    ax.axhline(_E_RHF_REF, color="C0", linestyle=":", alpha=0.5, label=f"RHF exact = {_E_RHF_REF:+.4f}")

    ax.set_xlabel("softening ε (Bohr)")
    ax.set_ylabel("helium ground-state energy (Hartree)")
    ax.set_title("Helium softening extrapolation, N=64 L=10, order=4")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"\nwrote figure {out_path}")


def _report(results, reference: float):
    print(f"\nRHF extrapolation (reference {reference:+.4f} E_h):")
    for model in ("linear", "quadratic", "linear_quadratic"):
        E0, _ = extrapolate_to_zero(results, model=model)
        err = E0 - reference
        print(
            f"  model={model:18s}  E₀ = {E0:+.5f} Ha   "
            f"residual = {err*1e3:+7.2f} mHa"
        )
    E0_best, _ = extrapolate_to_zero(results, model="linear_quadratic")
    return E0_best


def main():
    parser = argparse.ArgumentParser(description="Helium E(ε) → E(0) extrapolation")
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--L", type=float, default=10.0)
    parser.add_argument("--order", type=int, choices=[2, 4], default=4)
    parser.add_argument("--mixing", type=float, default=0.7)
    parser.add_argument("--max-iters", type=int, default=40)
    parser.add_argument("--lanczos-iters", type=int, default=40)
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=None,
        help="override the default ε sweep (in Bohr)",
    )
    parser.add_argument("--plot", action="store_true", help="save figure to docs/figures/")
    args = parser.parse_args()

    gato.enable_x64()

    h = args.L / args.N
    if args.epsilons is None:
        # Sweep around the default ε = h/2, covering a factor of ~3 in ε so the
        # linear/quadratic fit is well conditioned.
        epsilons = [0.25 * h, 0.375 * h, 0.5 * h, 0.75 * h, 1.0 * h]
    else:
        epsilons = args.epsilons

    print(f"Helium softening extrapolation   N={args.N}, L={args.L}, h={h:.4f}, order={args.order}")
    print(f"ε sweep (in units of h): {[f'{e/h:.3f}' for e in epsilons]}")
    print()

    print("RHF sweep:")
    rhf_results = run_sweep(
        args.N, args.L, epsilons,
        args.mixing, args.max_iters, args.order, args.lanczos_iters,
    )
    rhf_E0 = _report(rhf_results, _E_RHF_REF)

    if args.plot:
        out = Path("docs/figures/helium_softening_extrapolation.png")
        maybe_plot(rhf_results, rhf_E0, out)

    print()
    print("=" * 72)
    print(f"RHF    ε→0 = {rhf_E0:+.5f} Ha, residual vs reference = {(rhf_E0 - _E_RHF_REF)*1e3:+.2f} mHa")
    print(f"exact (non-rel, full CI):      {_E_EXACT:+.5f} Ha")


if __name__ == "__main__":
    main()
