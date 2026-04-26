# REWRITE.md — mission for the next session

This doc is a self-contained briefing for the model that opens the next
session. The README.md as it stands today documents seven phases (H →
H₂⁺ → He → H₂O → DFT → spectra → relativity), but the codebase no
longer matches: the trim landed in commits `0bca07a6d0ae` and `99331ae3f125`
removed Phase 5, 6, and 7 entirely, plus the neural-ansatz scaffolding
that was supporting them. The README now overstates scope and undersells
the new thesis. The mission is to rewrite it.

This file gives you everything you need: what changed, what the new
thesis is, what the new phase structure looks like, what the `qato`
sister project is and where it should be referenced, and a concrete
prompt at the bottom to copy-paste into the next session.

---

## 1. What changed in the codebase

**Four commits since the previous README state:**

```
6466d5f4f442 physics/water: Phase 4 H2O RHF + geometry driver [UNTESTED]
e46783672e7b pseudopotentials: HGH norm-conserving PPs (H, O) [UNTESTED]
99331ae3f125 physics/helium: Phase 3 RHF driver mirroring physics/hydrogen
0bca07a6d0ae trim: drop DFT, spectra, relativity, and neural ansatz; scope to RHF for H2O
```

**Removed entirely:**

| Module | Phase | Why removed |
|---|---|---|
| `src/gato/functionals.py` | 5 (DFT) | LDA + PBE deleted in favor of pure ab-initio RHF |
| `src/gato/spectra.py` | 6 | Excited-state machinery out of scope |
| `src/gato/physics/fine_structure.py` | 7 | Scalar-relativistic ZORA out of scope |
| `src/gato/ansatz/neural.py` | (cross) | Only consumer was a hydrogen demo path |
| `src/gato/solvers/vqe.py` | (cross) | Only consumer was the neural ansatz |
| `tests/test_{spectra,fine_structure,functionals,cusp}.py` | — | Tests for removed code |
| `benchmarks/radial_density.py` | — | Phase 1 visualization only |

**Truncated:**
- `scf.py`: KS-LDA path (`scf_ks_lda`, `ks_lda_energy`, `_KSFock`) deleted.
  Added an optional `V_nl_apply` callable parameter threaded through
  `RHFFock`, `_MixedFock`, `rhf_energy`, and `scf_rhf` so HGH non-local
  projectors plug in without further SCF surgery.
- `physics/hydrogen.py`: dropped the `vqe_neural` solver branch.
- Helium driver (now standalone, RHF-only).

**Added:**
- `src/gato/physics/helium.py` — Phase 3 driver mirroring the shape of
  `physics/hydrogen.py` (one-shot end-to-end function returning a
  `HeliumResult` dataclass with the converged orbital, density, full
  energy decomposition, virial diagnostic, orbital eigenvalues, history).
- `src/gato/pseudopotentials.py` — HGH norm-conserving pseudopotentials,
  analytic form, tabulated parameters for H and O. **UNTESTED.**
- `src/gato/physics/water.py` — Phase 4 driver: HGH-PP RHF on H₂O with
  geometry optimization via `jax.grad` on the BO total energy and Adam
  on nuclear positions. **UNTESTED.**

**Pedagogical framing changes (docstring-only):**
- `physics/h2_plus.py`: now explicitly framed as the Phase 2 bridge
  between Phase 1 (H) and Phase 3 (He) — one electron, multiple nuclei,
  reuses the GATO core unchanged.
- `physics/radial_hydrogen.py`: now explicitly framed as the
  "independent oracle" — a 1D log-radial bare-Coulomb solver kept *only*
  to cross-check the 3D Cartesian softening extrapolation. Not a
  production solver. Not used by any phase ≥ 1 driver.
- `scf.py`: header now states that RHF is the single SCF method and
  why. This is load-bearing for the new thesis.

**`IDEAS.md`:** the `qato` section was added in earlier conversation —
it's the design spec for the sibling VMC project that recovers the
correlation energy GATO can't see by construction. It also has a
"Water on a gamer GPU" section motivating the HGH pseudopotential
choice, and a Pallas-kernel section as a future performance pillar.
Don't move these; reference them from the README's roadmap section.

---

## 2. The new thesis

**The whole point of GATO is now: "more is different."** Anderson
(1972) — emergent behavior from many-body physics. Specifically:

> Solve Schrödinger from first principles → add atoms → emergent
> chemistry. No empirical input, no fitted XC functionals, no basis-set
> magic. Three ingredients only: Schrödinger, Pauli antisymmetry
> (Slater determinant), mean-field (Hartree–Fock).

The headline demo is **H₂O geometry from scratch**:

1. Place O + 2H at random positions.
2. Run RHF + Hellmann–Feynman forces + Adam.
3. Watch the bonds form (~0.94 Å), the angle settle (~106°), the lone
   pairs localize on oxygen, the dipole moment emerge (~2.0 D).

Nothing in that list was input by hand. **Chemistry as a consequence of
Schrödinger + Pauli + variational minimization.** That's the thesis.

The ~10% quantitative gaps (angle 106° vs experiment 104.5°; binding
energy ~9 eV vs experiment 10.1 eV) are exactly the electron-correlation
energy that mean-field cannot see by construction. This is *not a
weakness of the demo* — it's the precise place where the sister project
`qato` (VMC) becomes essential. The narrative arc:

- **GATO** — ab-initio mean-field. Where chemistry emerges.
- **qato** — VMC with explicit correlation. Where dispersion and the
  remaining ~10% emerge.

Two projects, one thesis: more = atoms + electrons + the right
wavefunction structure, with no empirical input.

### Why dropping DFT made the thesis sharper, not weaker

The old README told a seven-phase story where Phase 4 (RHF) and Phase 5
(DFT) sat side by side. That muddied the "ab initio" claim, because LDA
and PBE have empirical content (LDA fitted to the uniform electron gas;
PBE has tuned constants). With DFT in the mix, a careful reader can
fairly ask "how much of the result came from the empirical XC fit?"

**Pure RHF has zero empirical content.** Schrödinger, Pauli, mean-field.
That's it. Every emergent property — geometry, hybridization, polarity,
lone pairs, atomization energy — comes from those three and nothing
else. The "more is different" claim becomes the cleanest possible
sentence:

> "I imposed antisymmetry on a many-electron wavefunction, minimized the
> energy variationally, and water came out shaped like water."

The README rewrite should put this sentence on the box.

---

## 3. The new phase structure

```
Phase 1 — Hydrogen          (1 electron, 1 nucleus)
Phase 2 — H₂⁺               (1 electron, 2 nuclei) — bridge, optional in narrative
Phase 3 — Helium            (2 electrons, 1 nucleus, RHF)
Phase 4 — Water             (8 valence electrons, 3 nuclei, RHF + HGH PP) ← HEADLINE
                            geometry from jax.grad
```

**Each phase introduces exactly one new ingredient:**

| Phase | New ingredient |
|---|---|
| 1 | one electron in a nuclear potential |
| 2 | multiple nuclei (multi-center potential) |
| 3 | electron–electron interaction via mean field (Hartree + exchange) |
| 4 | molecular geometry from `jax.grad` on the SCF energy + pseudopotentials |

Beyond Phase 4, the README points the reader at `qato` for correlation
(He–He dispersion is the canonical demo).

### What used to be Phase 5/6/7 should not be revived

The old README's Phase 5 (DFT), Phase 6 (atomic spectra), and Phase 7
(scalar-relativistic heavy atoms) are deliberately dropped. Don't
include them as "future work" or "stretch goals" — they were
explicitly cut to sharpen the thesis. If a reader asks about
correlation, point at qato. If a reader asks about heavier atoms,
point at the limits of mean-field at d-shells (out of scope). If a
reader asks about excited states, the honest answer is "not the demo
this code is built for; use a TDDFT package."

---

## 4. The README rewrite — concrete checklist

The current `README.md` is large (~30k tokens) and most of the volume is
the §5 Roadmap section and the deep-dive phase notes. The rewrite
should:

1. **Keep §0–§2** mostly intact:
   - §0 "What GATO is" — update to single-thesis framing ("more is
     different" + RHF + ab initio + no empirical input).
   - §1 "Motivation" — keep as-is; the matrix-free / differentiable /
     JIT story is unchanged.
   - §2 "Physics background" — keep as-is; it's foundation material
     that the new structure also needs.

2. **Replace §3 "What's implemented"** with a current snapshot:
   - Phase 1 H driver (`gato-hydrogen`): converges to −0.5 E_h.
   - Phase 2 H₂⁺ driver (`gato-h2plus`): geometry to ~1.997 Bohr.
   - Phase 3 He driver (new, no CLI yet): RHF, ~−2.6 E_h on softened
     grid, ε→0 extrapolation closes to −2.86 E_h reference.
   - Phase 4 H₂O driver (`gato-water`): **CURRENTLY UNTESTED.** State
     this honestly — the code is in but the math hasn't been validated
     yet. Don't claim numbers you can't reproduce.

3. **Rewrite §5 "Roadmap"** as four phases, not seven. Drop §5.6
   ("FermiNet handoff") because the natural correlation handoff is now
   `qato`, not FermiNet.

4. **Drop §5.0's "Systems at a glance"** in its current form (it has
   rows for Phase 5/6/7) and rebuild it as a four-row table:

   ```
   | Phase | System | Method | Status |
   |---|---|---|---|
   | 1 | H, He⁺, Li²⁺ | imag-time on softened Coulomb | done |
   | 2 | H₂⁺ | imag-time + jax.grad geometry | done |
   | 3 | He | RHF closed-shell SCF | done |
   | 4 | H₂O | RHF + HGH PP + geometry | UNTESTED, wired |
   ```

5. **Add a new top-level section** introducing `qato` as the sibling
   project. Three or four paragraphs:
   - Where mean-field stops working (He–He dispersion is the canonical
     demo).
   - The VMC machinery (Slater–Jastrow trial, Metropolis sampling,
     local energy via `jax.grad`/`jax.hessian` of `log psi`).
   - That qato is documented in IDEAS.md as a separate project, not
     part of GATO's surface.

6. **Update §6 "Dependencies per phase"** to match the four-phase
   structure. Drop rows for the deleted phases.

7. **Strip §8 "References"** of the spectra / relativity citations.
   Keep the GATO-relevant ones.

8. **Rewrite §9 "Status"** to reflect current commits and the UNTESTED
   state of Phase 4.

9. **Audience and voice unchanged.** The README is still aimed at a
   physics student who has seen Griffiths but not a graduate course on
   computational electronic structure. Keep the "explain what, then
   explain why each design choice was made" pattern.

10. **Length target:** the rewritten README should be roughly 60–70%
    of the current size. Most of the savings come from collapsing
    Phase 5/6/7 prose (which is gone now) and tightening the four
    remaining phase notes.

### A specific word about IDEAS.md and qato

Don't fold the qato spec into the README — it's a separate project's
design doc. Reference it like this:

> **For the correlation half of the story** — what mean-field cannot
> see, and how to recover it from a Slater–Jastrow VMC trial on
> He–He — see [`IDEAS.md`](IDEAS.md), section "qato — a sibling
> project for Quantum Monte Carlo".

This keeps GATO's scope clean while pointing readers at the natural
follow-up.

---

## 5. Open issues to surface honestly

The rewritten README should not paper over these:

1. **Phase 4 is wired but not validated.** `pseudopotentials.py` and
   `physics/water.py` exist (commits `e46783672e7b` and `6466d5f4f442`)
   but neither has been run end-to-end. The HGH math has not been
   smoke-tested against published numbers. The water geometry
   optimization has not been demonstrated to converge to the right
   shape. Future commits will validate; until they do, the README
   should mark Phase 4 as "scaffolding in place; numbers pending."

2. **No HGH parameters for elements beyond H and O.** Adding C, N, F
   takes ~20 lines of tabulated values per element, but it's not done.
   The README's "Systems at a glance" should not promise systems we
   can't currently run.

3. **The Pallas-kernel speedup is still in IDEAS.md as a design.** Don't
   claim Pallas wins in the README. The pure-JAX stencil is the
   currently shipping kinetic operator.

4. **Geometry-optimization performance is unknown.** Targeted: ~minutes
   per H₂O geometry on a consumer GPU. Reality: not yet measured. State
   the target, mark "to measure."

---

## 6. Prompt for the next session

Copy-paste the block below into a fresh session in this repo. It
references this file, so the next-session model can read REWRITE.md
plus the current README.md plus the codebase and produce the rewrite.

---

```
You are taking over a GATO codebase rewrite mid-flight. The previous
session trimmed the codebase to focus on a single thesis — "more is
different" via ab-initio RHF on H₂O — but the README.md still describes
the old seven-phase structure. Your job is to rewrite README.md so it
matches the current code.

Start by reading these in order:

1. REWRITE.md — full briefing for this mission. Read it first.
2. The current README.md — the document you'll be replacing.
3. IDEAS.md — for the qato sibling-project context that the rewrite
   needs to reference.
4. src/gato/scf.py header docstring — concise statement of the new
   single-thesis scope.
5. src/gato/physics/{hydrogen,helium,h2_plus,water}.py docstrings —
   each phase driver's pedagogical role is now made explicit at the
   top of each file. The README should align with what those say.

Then produce a single revised README.md, replacing the existing one.
Constraints:

- Audience and voice: physics student who has seen Griffiths;
  "explain what, then explain why" pattern.
- Phase structure: exactly four phases (H, H₂⁺, He, H₂O). No DFT,
  no spectra, no relativity.
- Thesis: "more is different" — Anderson 1972. Chemistry emerges
  from Schrödinger + Pauli + mean-field with no empirical input.
- Honest status: Phase 4 (H₂O) is wired but UNTESTED. State that
  plainly. The code in physics/water.py and pseudopotentials.py
  has not been validated; their commits are tagged [UNTESTED].
- Reference qato (the VMC sister project) as the natural follow-on
  for correlation; do not fold its content into GATO's README.
- Length: aim for ~60–70% of current size.
- Keep §1 (Motivation) and §2 (Physics background) substantively
  intact — they're foundation material that survives unchanged.

Produce the rewrite. Don't ship a draft for review — write the
final README.md you'd be willing to put in front of a reader.

After the rewrite, run a quick sanity pass:
  - grep for any remaining references to Phase 5/6/7, "DFT", "LDA",
    "PBE", "ZORA", "spectra", "FermiNet", "neural ansatz", "VQE".
    These should all be gone or explicitly marked as out-of-scope.
  - Verify all module references in the README correspond to files
    that exist in src/gato/.
  - Verify all CLI commands referenced exist in pyproject.toml.

Then summarize what changed in 5–8 bullets.
```

---

## 7. What I would *not* do in the rewrite

A few things to leave alone, because they're easy to overcorrect:

- **Don't drop §1 (Motivation).** The "matrix-free + differentiable + JIT"
  story is unchanged and remains the strongest opening argument.
- **Don't rewrite §2 (Physics background).** It's foundation material.
  Tweaks for clarity are fine; structural rewrites aren't necessary.
- **Don't oversell H₂O.** The numbers aren't validated yet. Saying
  "GATO computes water's geometry to within 2% of experiment" is a
  promise we can't currently keep. Say "GATO is wired to compute
  water's geometry; validation pending" until the smoke tests land.
- **Don't reintroduce the FermiNet handoff.** The natural follow-on for
  correlation is now `qato`. FermiNet is referenced inside the qato
  IDEAS section as a more flexible trial wavefunction within VMC; it's
  not a separate handoff at the GATO level anymore.
- **Don't add a "future work" section.** The trim was the point. If
  something isn't in the four phases, it isn't a stated goal.

---

## 8. Definition of done

The README rewrite is done when:

- Reading the new README front-to-back, a physics student can identify
  the four phases, understand what each one demonstrates, and reach the
  end with a clear answer to "what does GATO do, and what doesn't it
  do?"
- A reviewer running `grep -E "DFT|LDA|PBE|ZORA|spectra|fine.structure|FermiNet|neural.*ansatz|vqe"`
  on the new README finds nothing (or only deliberate "out of scope"
  mentions).
- Every CLI command, file path, and module mentioned in the README
  exists in the current tree.
- The qato sister project is referenced in exactly one place (a single
  short section pointing at IDEAS.md) — not folded in.
- The honesty markers on Phase 4 (UNTESTED, validation pending) are
  explicit.
- Length is roughly 60–70% of the current README.
