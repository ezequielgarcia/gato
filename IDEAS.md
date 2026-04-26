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

## Pallas kernel for the $H\psi$ inner loop

### Framing

Every solver we've built — imag-time, Lanczos, Rayleigh-quotient descent,
the SCF Fock-apply — bottoms out in the same operation: apply $\hat H = \hat T + V$
to a $\psi$ array on the 3D grid, thousands of times per solve. Today this
is pure JAX: `jnp.pad` / `jnp.roll` for the 7-point (or 13-point) stencil,
a pointwise $V\psi$, and a reduction for the Rayleigh quotient. Correct and
reasonably fast, but bandwidth-bound and touches the $N^3$ array ~4× per
apply.

A [Pallas](https://jax.readthedocs.io/en/latest/pallas/index.html) kernel
(JAX's Triton-lowering DSL) fuses the stencil + $V\psi$ + reduction into a
single launch, tiling the cube into shared memory so each interior point
is read once. Classic GPU stencil territory.

### Scope

GPU-only (NVIDIA via Triton); keep the existing JAX path as the fallback,
dispatch on `jax.default_backend()`. TPU has Pallas support too but would
need a separate Mosaic kernel; out of scope.

### First PR

1. `pallas_laplacian_plus_v(psi, V, h)` — 7-point stencil + local potential,
   Dirichlet BC, fp32/fp64 parametric. ~80–120 lines.
2. Parity test against `operators.laplacian` + `hamiltonian.apply` to
   $10^{-10}$ on a $32^3$ random input.
3. Micro-benchmark on $96^3$ and $128^3$, fp32 and fp64.
4. Wire behind `Hamiltonian(..., use_pallas=False)` so nothing downstream
   changes by default.

### Follow-ups (each its own PR)

- Fuse $\langle\psi|\hat H|\psi\rangle$ reduction into the same kernel —
  biggest win for Lanczos/imag-time since the stencil output never needs
  to materialize.
- 4th-order 13-point stencil variant.
- ZORA kinetic: same stencil shape with position-dependent
  $K(\mathbf r) = c^2 / (2c^2 - V(\mathbf r))$ loaded per-point; also fuses
  the $K'/r$ correction term that Phase 7 needs.
- Fused $1/k^2$ multiply on the inverse-FFT output in `solvers/poisson.py`.

### Expected payoff

Bandwidth-bound on a single consumer GPU, so realistic speedup on the
stencil itself is ~1.5–3× rather than 10×. The bigger win is composition:
Lanczos/imag-time do $O(100\text{–}1000)$ applies per solve, and the SCF
loop then does tens of those. At fp32 the speedup stacks; at fp64 on
consumer NVIDIA parts (1/32 throughput of fp32) the kernel is still
bandwidth-bound so the proportional win is similar.

### Caveats

- The stock JAX stencil is already pretty good — worth measuring before
  claiming numbers.
- Double precision on consumer NVIDIA GPUs is 1/32 the fp32 throughput;
  if GATO is fp64 everywhere the compute-side ceiling is lower than a
  naïve roofline suggests.
- Pallas kernels don't yet compose with `jax.grad` as cleanly as pure
  JAX — if autodiff through the Hamiltonian matters for a solver (Pulay
  forces, differentiable PP parameters), keep the JAX path for those.

### Where *not* to put a Pallas kernel

- `solvers/imag_time.py` FFT split-operator step — cuFFT saturates BW.
- `ansatz/neural.py` MLP — XLA fuses dense + activations fine.
- 1D radial solvers in `physics/radial_hydrogen.py`, `physics/fine_structure.py`
  — $N \sim 10^3$, kernel launch overhead dominates.
- `scf.py` DIIS / mixing — orchestration, not arithmetic.

---

## `qato` — a sibling project for Quantum Monte Carlo

### Framing

GATO is structurally mean-field: one-electron orbitals on a grid, optimized variationally. That's why it teaches Schrödinger-on-a-grid beautifully but is **structurally blind to dispersion** (London R⁻⁶ attraction needs correlated motion of electrons on different atoms; mean-field factorizes that away). The He–He dimer is the canonical demonstration: pure HF gives a monotonic repulsive wall, no minimum, because dispersion is a correlation effect that mean-field cannot see by construction.

The right next step beyond Phase 5 is not "more grids, finer spacing" — it's **methods that respect antisymmetry and capture correlation simultaneously**. Quantum Monte Carlo (VMC + DMC) is the cleanest pedagogical entry point into that world.

### Why a separate repo

VMC's primitives — electron-position walkers, log-ψ evaluation, local-energy estimator, Metropolis moves, branching populations — share almost nothing with GATO's grid-and-stencil core. Mixing them would pollute GATO's clean grid-centric design with stochastic-sampling abstractions that have a different shape entirely. What's reused is small but valuable: JAX/jit infrastructure, atomic units, the converged Phase-3 He reference for benchmarking, and Hamiltonian operators expressed as pure functions.

Suggested name: **`qato`** (parallels GATO; signals lineage as a sibling, not a replacement). **Scope: VMC only.** DMC is explicitly out of scope for v1 — it doubles the code volume, adds several new failure modes (branching, fixed-node, time-step extrapolation, population control), and its pedagogical lesson (imaginary-time projection + fixed-node theorem) is meta-methodology rather than the headline physics. The headline physics — correlation, dispersion, antisymmetric wavefunctions — is fully captured by VMC.

### Phase progression (mirrors GATO's incremental shape)

- **Phase 0 — H (infrastructure, not physics).** Slater 1s with $\zeta = 1$ is the *exact* hydrogen ground state, so VMC on it has zero variance — local energy $E_L = \hat H\psi/\psi$ is constant at $-0.5\,E_h$ everywhere. Use this phase for infrastructure: Metropolis sampler, walker data structure, local-energy estimator via `jax.grad`/`jax.hessian` of $\log\psi$, autocorrelation diagnostics, blocking analysis for error bars. Then deliberately break it — start with a Gaussian trial $e^{-\alpha r^2}$ instead of $e^{-\zeta r}$ and watch VMC optimize $\alpha$ while never reaching $-0.5$ exactly (wrong cusp). The **zero-variance principle** is the lesson.
- **Phase 1 — Hydrogenic (Z-scaling, excited states).** He⁺, Li²⁺: same code, different $Z$. Adds the $\ell > 0$ excited-state machinery (Slater determinant of orthogonal one-electron orbitals). Validates that the sampler handles nodes correctly — 2p has a nodal plane, walkers must cross it cleanly.
- **Phase 2 — He VMC (correlation appears).** Slater–Jastrow trial: determinant of two 1s orbitals × $\exp[u(r_{12})]$ with a 2–4 parameter Padé. HF gives $-2.862\,E_h$, exact is $-2.903\,E_h$. Watch Jastrow optimization close ~90% of that gap. **The Jastrow factor is where electron correlation lives, made visible.**
- **Phase 3 — He–He (dispersion emerges).** Two He nuclei at separation $R$, four electrons, Slater–Jastrow generalized to inter-atomic Jastrow terms $u(r_{ij})$ between electrons on different atoms. Sweep over $R$, subtract $2 E(\text{He})$, plot. **The R⁻⁶ tail emerges from the optimized inter-atomic Jastrow; the repulsive wall emerges from the determinant.** Both at once, no hand-tuning, no empirical input. Terminal demo of the project. Quantitative agreement with experiment is *not* the goal — qualitatively-correct shape (well depth within a factor of ~2, equilibrium $R$ within ~10%) is sufficient and is what VMC alone delivers via systematic error cancellation between dimer and atomic limits.

### Out of scope (deferred indefinitely, possibly forever)

- **DMC.** Listed here for completeness, not as a roadmap item. Would be the natural next step if quantitative experimental agreement ever became the goal — branching walkers (or fixed-population reconfiguration), fixed-node projection, time-step extrapolation, population-control bias correction. Adds ~2–3× code volume; the headline physics is already done at VMC. Revisit only if there's a concrete reason (paper, benchmark, comparison to literature numbers).
- **Backflow and neural-network trial wavefunctions.** A natural progression *within* VMC — replace the hand-designed Slater–Jastrow with backflow coordinates, then with a FermiNet-style network. Each step is a more flexible parameterization optimized by the same VMC machinery. Out of scope for v1 because Slater–Jastrow is enough to demonstrate dispersion on He–He; revisit if extending to larger systems or aiming at neural-VMC benchmarks.

### Pedagogical pacing

In GATO the H phase is genuinely a payoff ("we computed hydrogen from a 3D grid!"). In QMC the H phase is *infrastructure* ("we built a sampler") because the interesting QMC physics — correlation — only appears at He. Worth signposting in the README: Phase 0/1 are scaffolding, Phase 2 is where the satisfying physics starts.

### Minimum viable first PR

He₂ at one fixed separation, Slater–Jastrow trial, VMC only: ~400 lines.
- Slater determinant from frozen Phase-3 He 1s orbitals.
- Jastrow as 2–4 parameter Padé in $r_{ij}$.
- Local energy via `jax.grad`/`jax.hessian` of $\log\psi$.
- Metropolis sampling with autocorrelation diagnostics.
- Optimize Jastrow parameters by SR or Adam on the variance-reduced energy.
- The pedagogical money shot: $E_{\rm VMC} < E_{\rm HF}$. **That gap is the dispersion energy**, derived from first principles in your own code.

Defer DMC until VMC works on He₂ and reproduces a published number; DMC adds branching walkers, fixed-node projection, and population control (~2–3× the code volume), much harder to debug if VMC isn't already trustworthy.

### Tradeoffs vs GATO

- **Lose:** clean `jax.grad` of energy w.r.t. nuclear positions. VMC energy is a stochastic estimator with sampling noise; nuclear gradients need variance reduction (reweighting, correlated estimators, Hellmann–Feynman with reweighting). Different optimization regime than GATO's deterministic one.
- **Gain:** correlation captured by construction. Dispersion, static correlation, near-degeneracies — all visible. Antisymmetry is a hard constraint (determinant structure), not a soft penalty.
- **Architectural cost:** ~$N^3$ per sample × millions of samples per energy. He₂ is feasible on a laptop CPU; anything beyond ~10 electrons wants a GPU. The walker dimension parallelizes embarrassingly.

### Why this beats "add correlation to GATO"

Adding post-HF correlation (MP2, CCSD) on top of GATO's grid would require formulating those methods in real space, which essentially nobody does (they're naturally MO-basis methods with $O(N^7)$ tensor contractions). You'd be writing a Gaussian-basis quantum-chemistry code from scratch, which is not GATO's mission. QMC is **the** correlation method that lives natively in real space — same coordinate system as GATO, different sampling strategy.

### Aside: why R⁻⁶ is called "dispersion"

The name is borrowed from **optics**, and the connection is a derived mathematical identity rather than a casual analogy. In 1930, Fritz London showed that the R⁻⁶ attraction between two neutral atoms can be calculated from a single integral involving each atom's **frequency-dependent polarizability** $\alpha(\omega)$:

$$C_6 \;=\; \frac{3}{\pi}\int_0^\infty \alpha_A(i\omega)\,\alpha_B(i\omega)\,d\omega$$

(the **Casimir–Polder integral**, evaluated along the imaginary-frequency axis). The same $\alpha(\omega)$ determines the refractive index of a gas, $n(\omega) \approx 1 + 2\pi N\,\alpha(\omega)$, whose frequency dependence is called **optical dispersion** — the reason blue light refracts more than red, the reason rainbows exist. So the R⁻⁶ coefficient between atoms is literally an integral over the same dynamical polarizability that causes white light to split into colors.

That equivalence is the etymology. London called the force "dispersion" because its strength is computable from the *dispersion data* of the atom — its absorption spectrum, oscillator strengths, polarizability vs. frequency. Before London, van der Waals (1873) had postulated an empirical R⁻⁶ attraction with no derivation; London's 1930 paper showed two things at once: (1) the attraction comes from correlated quantum fluctuations of the electron clouds (the modern "instantaneous dipole–induced dipole" picture), and (2) the coefficient is fixed by the same atomic transition data you measure with a spectrometer. The optical-dispersion connection wasn't an afterthought — it was the proof that the force was real and quantitative.

**Why the optical connection isn't a coincidence.** Both phenomena come from the same underlying object: the atomic polarizability operator $\hat\alpha(\omega) = \sum_n \frac{|\langle 0|\hat\mu|n\rangle|^2 \cdot 2\omega_n}{\omega_n^2 - \omega^2}$ — a sum over excited states weighted by transition dipole moments. In optics, this controls how the atom responds to an *external* oscillating field (the photon). In the dispersion-force calculation, this controls how atom A responds to atom B's *own* fluctuating field, with the integral over all frequencies giving the binding energy. Same operator, same matrix elements, two physical settings filtered through different geometries.

**Terminology zoo, untangled:**
- **van der Waals force** — catch-all umbrella for all weak intermolecular forces. Three sub-types: Keesom (permanent dipole–permanent dipole), Debye (permanent dipole–induced dipole), London dispersion (correlated fluctuating dipoles).
- **London dispersion force** — the universal one. Exists between all pairs of atoms because every atom polarizes in response to fluctuations. The *only* one of the three that operates between two He atoms (no permanent moments → only fluctuation-correlation contributes). This is what `qato`'s He–He demo recovers.
- **Casimir / Casimir–Polder force** — same physics at longer range where retardation (finite speed of light between fluctuation and response) kicks in. Switches the asymptote from R⁻⁶ to R⁻⁷ at distances $\gtrsim$ atomic transition wavelength (~hundreds of nm).

The satisfying part of `qato`'s terminal demo: VMC will reproduce that R⁻⁶ from a Slater–Jastrow wavefunction with no spectral data input — recovering, from first principles, the same coefficient you could have looked up in a polarizability table. **Dispersion = the frequency dependence of polarizability = the data needed to compute the R⁻⁶ coefficient.** The force is named after how it's calculated, not after what it physically does.

### Connection to GATO's §5.6 FermiNet handoff

QMC is the conceptual bridge to FermiNet. The progression is: Slater–Jastrow VMC → backflow VMC → neural-network VMC (FermiNet/PauliNet). Each step replaces a hand-designed component with a more flexible parameterization, optimized variationally. A `qato` project that gets to Slater–Jastrow He–He puts the reader exactly one architectural step away from FermiNet, and the §5.6 handoff in the GATO README becomes a "and here's where this approach goes next" rather than a black-box reference.

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
