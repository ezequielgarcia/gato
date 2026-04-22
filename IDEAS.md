# IDEAS

Exploratory notes on methods beyond the current roadmap. Not commitments —
just sketches for discussion.

---

## Water on a gamer GPU, without neural VMC

### Framing

The current Phase 4 plan (all-electron RHF on a 3D Cartesian grid with
softened Coulomb) will struggle with water on an RTX 5070. Oxygen's 1s
orbital has characteristic scale ~1/8 = 0.125 $a_0$, so honest resolution
needs $h \lesssim 0.05\,a_0$, i.e. 256³ or larger at a 12 $a_0$ box. With
5 doubly-occupied orbitals, $n_\text{orb}^2$ exact-exchange Poisson solves
per SCF iteration, and a softening-extrapolation sweep on top, a single
geometry optimization step is hours. Geometry optimization itself: days.

The move is to **stop trying to resolve the oxygen core.**

### The headline answer: pseudopotentials + the existing architecture

Replace each $-Z_k/r$ with a **norm-conserving pseudopotential** (HGH / ONCV):

1. **Eliminates the cusp.** No softening parameter, no ε→0 extrapolation,
   no cancellation disasters at the nucleus. Smooth by construction.
2. **Removes core electrons.** Oxygen 8 → 6 valence. Water 10 → 8 electrons.
3. **Relaxes the grid.** Valence orbitals have scale ~1 $a_0$. Spacing
   $h \sim 0.3$–$0.4\,a_0$ is enough. 64³–96³ fits in ~100 MB at fp64.
4. **Keeps everything already built.** Uniform Cartesian grid, matrix-free
   stencil kinetic, FFT-based Hartree Poisson, LDA or exact exchange,
   `jax.grad` for Pulay forces. SCF loop unchanged.

HGH pseudopotentials are **analytic**: a Gaussian form plus a small set of
projectors, ~10 parameters per element, tabulated in Hartwigsen–Goedecker–
Hutter 1998. No interpolation tables, no external files — the whole H and O
pseudopotential is ~12 lines of JAX. The non-local projector term is a few
inner products per orbital per SCF iteration. Total code cost: ~150 LOC on
top of the existing stack.

### Expected performance on a 5070

- H₂O single-point RHF at $64^3$, ~8 valence orbitals, ~15 SCF iterations
  → few seconds.
- Geometry optimization (50 SCF solves with DIIS warm-starting) → under
  a minute.

Not a stretch — GPAW and Quantum ESPRESSO routinely do this on similar
hardware. For GATO this is a clean pedagogical addition: pseudopotentials
are a named textbook concept and implementing HGH in JAX from scratch
is a genuinely educational chunk of the Phase 4 writeup.

### Novel angles (if a paper is the goal, not just a result)

1. **Differentiable pseudopotential design.** Put HGH parameters under
   `jax.grad`. Train the pseudopotential for each element against a small
   all-electron reference (atom + dimer) such that the RHF+PP energy and
   force landscape matches all-electron RHF on a fine grid. Then freeze
   and use for production. Underexplored in the JAX era — Kasim & Vinko
   (2022) and jax-dft have scratched the surface; nobody has done it
   rigorously for a suite of elements.

2. **ΔML-corrected LDA.** Train a small MLP that takes $(\rho, |\nabla\rho|,
   \tau)$ at each grid point and outputs a correction to the LDA XC energy
   density. Reference data: a few hundred CCSD(T) energies from public
   datasets. Optimize end-to-end with the SCF in the loop — the network
   is called inside `jax.lax.while_loop`. Kirkpatrick et al. 2021 (DeepMind
   DM21) did a heavy version of this; a 5070-sized version for small
   molecules would be a clean publication. Water geometry at near-CCSD(T)
   accuracy at LDA cost.

3. **FFT-based exact exchange done properly.** The exact exchange operator
   is $n_\text{orb}^2$ Poisson solves. On a uniform grid with FFT this is
   $O(n_\text{orb}^2 \cdot N^3 \log N)$ — for water ~32 FFTs of $64^3$ per
   SCF iteration, a few ms each on a 5070. Practical, no approximation,
   and "exact exchange without Gaussian basis" is a meaningful
   methodological point.

4. **Grid-free Coulomb via ERI-in-a-box.** Evaluate Coulomb and exchange
   integrals against a Gaussian auxiliary basis whose exponents match the
   pseudopotential's valence shell, then grid-interpolate. Hybrid
   Gaussian/grid (GPW method, à la CP2K). More involved; probably not a
   two-week project.

### Ranking

| Approach | Time on 5070 | Accuracy for H₂O geometry | GATO effort | Novelty |
|---|---|---|---|---|
| Current plan (all-electron, softening) | hours–days | ~1° on angle | 0 | 0 |
| Pseudopotential + LDA on uniform grid | seconds–minutes | ~1° on angle | ~150 LOC | 0 (mature) |
| Pseudopotential + exact exchange (RHF) | ~1 min | ~1.5° off (RHF limit) | ~250 LOC | 0 (mature) |
| Pseudopotential + diff. optimization of PP params | minutes | target accuracy | ~400 LOC | genuinely novel |
| Pseudopotential + ΔML-corrected LDA | minutes | near CCSD(T) | ~600 LOC + training | very novel |
| Non-neural grid-based VMC (Slater-Jastrow) | hours | ~chemical accuracy | ~800 LOC | mildly novel (non-neural VMC in JAX) |

### Bottom line

- **If the goal is "water, geometry, fast, on a 5070, no NN-VMC, pedagogically
  respectable":** HGH pseudopotentials on the existing uniform-grid real-space
  SCF stack. Two weekends of work, 100× speedup, geometry in under a minute.
  Slots cleanly between Phase 3 and Phase 4 — effectively a new "Phase 3.5:
  pseudopotentials."

- **If the goal is a publishable novel contribution:** stack differentiable
  pseudopotential optimization on top. Nobody has a JAX-native,
  end-to-end-differentiable pseudopotential generator coupled to a molecular
  SCF solver on consumer hardware.

- **What I would not do:** try to solve all-electron water on a real-space
  Cartesian grid without softening. That's the route where cheap GPU
  hardware hits a wall and days are spent on numbers GPAW produces in
  seconds. The pseudopotential decision is what separates "GATO can do
  water" from "GATO cannot do water" on a 5070.

---

## Log-radial cross-check for Phase 1 (implemented)

See `src/gato/physics/radial_hydrogen.py`. Standalone 1D log-radial solver
with pure $V = -Z/r$, as an independent validation of the $\epsilon \to 0$
extrapolation used by the 3D Cartesian stack. Reproduces $-Z^2/2$ to
$\sim 10^{-7}\,E_h$ at $N = 800$ for $Z \in \{1, 2, 3\}$. Does not
generalize beyond single-nucleus $\ell = 0$; kept strictly as a sanity
check on the Phase 1 number.

---

## Rejected / evaluated ideas (record of discussion, not plans)

- **Log-radial as the main 3D grid.** Doesn't generalize to molecules (can't
  have multiple origins); Phase 1 ε → 0 extrapolation already gets within
  1%. Rejected for the 3D stack. Kept as 1D cross-check only.

- **Kustaanheimo–Stiefel transformation for neural VMC.** Genuinely novel,
  mathematically elegant, but: (a) gauge redundancy of the 4D → 3D map
  requires $U(1)$-equivariant architectures or wasteful symmetrization;
  (b) the eigenvalue enters the transformed potential, breaking the clean
  Rayleigh-quotient gradient flow; (c) most importantly, the transformation
  doesn't compose across multiple nuclei, so it solves only the single-atom
  case — which Kato cusp factors already handle and then generalize to
  molecules. Rejected for production. Worth considering as a single-atom
  pedagogical experiment to document the Duru–Kleinert / 4D-oscillator
  picture.
