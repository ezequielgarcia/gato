# GATO Phase 6: Atomic absorption spectra — foundations

> **Read this on <https://ezequielgarcia.github.io/gato/phase6_note/>** — GitHub's markdown viewer does not render display math reliably.

**Companion note to `phase1_paper.md`, `phase2_note.md`, and `phase3_note.md`.** Phase 6 is the first phase that goes *beyond ground states*: the primitives for extracting atomic absorption spectra from excited-state eigenpairs and dipole matrix elements, applied first to hydrogen on the log-radial grid of Phase 1. Multi-electron drivers, GGA cross-sections, and the end-to-end `physics/atomic_spectra.py` script remain future work; this note documents the pieces that are in place and numerically validated.

## Abstract

Phase 6 adds four operator-level primitives — `transition_dipole`, `oscillator_strength`, `einstein_A`, `photon_wavelength_nm` — plus the nonzero-$\ell$ extension of the log-radial solver built in Phase 1 (`solve_bound_states(grid, Z, K, \ell)`) and the ℓ-independent radial factor `radial_dipole(u_f, u_i, grid)` of any electric-dipole matrix element. On an $N = 800$ log-radial grid the Balmer H-α, H-β, and H-γ line positions are recovered to $< 0.1\,\text{nm}$ vs the infinite-nuclear-mass Rydberg prediction — the Phase 6 exit criterion for hydrogen. The analytic $\langle 1s | z | 2p_z\rangle = 128\sqrt 2/243$ is independently reproduced on a Cartesian grid via `transition_dipole` and on the log-radial grid via `radial_dipole` composed with the angular Gaunt factor $1/\sqrt 3$, confirming the two stacks are consistent. The textbook $f(1s \to 2p) = 0.41620$ comes out of `oscillator_strength`.

## 1. Scope

Phase 6 is framed as an observable-extraction layer on top of the Phase 1 and Phase 3 solvers rather than a new solver phase. The Lanczos machinery from Phase 1 already returns the lowest $K$ eigenpairs of any matrix-free Hermitian operator; Phase 6 uses those eigenpairs to build derived observables (line positions, oscillator strengths, emission rates, natural linewidths). The primitives are small: each is a closed-form function of a precomputed wavefunction pair and a frequency, in atomic units throughout ($\hbar = m_e = e = 1$; $c = 1/\alpha \approx 137.036$).

The scope of this note is the numerical *primitives* and the hydrogen line positions obtained from ℓ=0 radial eigenvalues. Numerical oscillator strengths for specific multi-electron transitions (He $1s^2 \to 1s2p$), full absorption cross-sections with Doppler + Lorentzian broadening, and the `physics/atomic_spectra.py` end-to-end driver are not covered; they are listed in `README.md` §5.6 as remaining work.

## 2. Transition dipole on a real-space grid

For wavefunctions $\psi_a(\mathbf r)$ and $\psi_b(\mathbf r)$ on the cell-centered Cartesian grid of Phase 1, the electric-dipole matrix element is computed directly as a grid sum:

$$
\boldsymbol{\mu}_{a \leftarrow b}
\;=\; \langle \psi_a | \hat{\mathbf r} | \psi_b\rangle
\;=\; \int \psi_a^{*}(\mathbf r)\,\mathbf r\,\psi_b(\mathbf r)\,dV
\;\approx\; h^3 \sum_{ijk} \psi_{a,ijk}^{*}\,(X, Y, Z)_{ijk}\,\psi_{b,ijk}. \tag{1}
$$

The implementation `gato.spectra.transition_dipole(psi_a, psi_b, grid)` returns the complex 3-vector $(\mu_x, \mu_y, \mu_z)$; `grid.coords()` supplies $X, Y, Z$. No integration-by-parts, no approximation of the position operator — the dipole is a pointwise product and a midpoint-rule quadrature, matching the Phase 1 volume element $dV = h^3$ exactly.

### 2.1 Analytic cross-check

For hydrogen 1s ($\psi_{1s} = \pi^{-1/2} e^{-r}$) and 2p$_z$ ($\psi_{2p_z} = (4\sqrt{2\pi})^{-1} z\,e^{-r/2}$), the nonzero Cartesian component is

$$
\mu_z \;=\; \langle \psi_{1s} | z | \psi_{2p_z}\rangle
\;=\; \frac{1}{4\pi\sqrt 2}\,\frac{1}{\sqrt\pi}\,\int z^2\,e^{-r/2}\,e^{-r}\,d^3r
\;=\; \frac{128\sqrt 2}{243} \;\approx\; 0.74490\,a_0, \tag{2}
$$

by Bethe–Salpeter §63. `tests/test_spectra.py::test_transition_dipole_1s_2pz_hydrogen` evaluates eq. (1) with the analytic 1s and 2p$_z$ on an $80^3$, $L = 20\,a_0$ grid and confirms the three components of $\boldsymbol\mu$: $|\mu_x|, |\mu_y| < 10^{-10}$ (parity; $x$ and $y$ are odd under the relevant reflection and the integrand has even $x$, $y$ behavior), $\mu_z$ agrees with eq. (2) to better than $10^{-2}$.

## 3. Oscillator strength and Einstein A

In atomic units the dimensionless oscillator strength for an electric-dipole transition is

$$
f_{0\to e} \;=\; \frac{2}{3}\,\omega_{e0}\,|\boldsymbol\mu_{e\leftarrow 0}|^2, \tag{3}
$$

with $\omega_{e0} = E_e - E_0$ the transition frequency. `gato.spectra.oscillator_strength(omega, mu)` is a two-line function of $\omega$ and the dipole 3-vector. The textbook $f(1s \to 2p) = 0.41620$ is recovered as

$$
f_{1s \to 2p} \;=\; \sum_{m = -1}^{+1} \frac{2}{3}\,\omega_{1s\to 2p}\,|\mu^{(m)}|^2
\;=\; 2\,\omega_{1s\to 2p}\,|\mu_z|^2, \tag{4}
$$

because the three 2p$_m$ sublevels contribute equal $|\mu|^2$ by symmetry, and $\omega_{1s\to 2p} = E_{2p} - E_{1s} = 3/8$ Hartree for hydrogen. Numerically: $2 \cdot 0.375 \cdot (128\sqrt 2/243)^2 = 0.41620$.

The spontaneous emission rate (Einstein $A$) follows from

$$
A_{e\to 0} \;=\; \frac{4\,\omega_{e0}^3}{3\,c^3}\,|\boldsymbol\mu_{e\leftarrow 0}|^2, \tag{5}
$$

with $c = 1/\alpha \approx 137.036$ in atomic units. The natural linewidth $\Gamma_{\rm nat} = \hbar\,A$ is numerically equal to $A$ in atomic units. `einstein_A(omega, mu)` provides both scaling checks (factor-8 response to doubled $\omega$) and the numerical value used downstream for line broadening.

## 4. ℓ > 0 channels of the log-radial solver

Phase 1's log-radial solver `physics/radial_hydrogen.py` originally handled the ℓ = 0 channel only. Phase 6 extends it to arbitrary $\ell$ by adding the centrifugal barrier to the ℓ=0 radial Hamiltonian,

$$
\hat H_\ell \;=\; -\tfrac{1}{2}\,\frac{d^2}{dr^2} \;+\; \frac{\ell(\ell+1)}{2 r^2} \;+\; V(r), \tag{6}
$$

acting on $u_{n\ell}(r) = r\,R_{n\ell}(r)$. For hydrogen ($V = -Z/r$), eigenvalues in every $\ell$-channel collapse onto $E_n = -Z^2/(2 n^2)$ with $n \ge \ell + 1$ — the accidental ℓ-degeneracy of the Coulomb problem.

### 4.1 Boundary at $r = 0$: parity depends on ℓ

Near the nucleus, $u(r) \sim r^{\ell+1}$. Reflecting $u(-r) = (-1)^{\ell+1}\,u(r)$ makes the 4th-order chain-rule Laplacian stencil see a smoothly-extended function instead of an artificial jump. The padding helper `_pad_boundary(u, width, left_sign)` with

$$
\text{left\_sign} \;=\; (-1)^{\ell + 1}
\;=\; \begin{cases} -1 & \ell = 0, 2, 4, \ldots \\ +1 & \ell = 1, 3, 5, \ldots\end{cases} \tag{7}
$$

preserves the 4th-order accuracy of the stencil at the innermost grid points for any $\ell$. Using the wrong parity or zero-padding there biases the kinetic energy by $O(h^2)$ and spoils grid convergence.

### 4.2 Verification

`tests/test_radial_ell.py` covers three invariants:

1. Eigenvalues in the $\ell = 1$ (2p, 3p, 4p), $\ell = 2$ (3d, 4d, 5d), and $\ell = 3$ (4f, 5f, 6f) channels all reproduce $E_n = -Z^2/(2 n^2)$ to $10^{-4}\,E_h$ at $N = 800$.
2. $E_{2s}$ (from ℓ=0) and $E_{2p}$ (from ℓ=1) agree to $10^{-4}\,E_h$, reproducing the Coulomb ℓ-degeneracy.
3. The ℓ=0 path still recovers $E_0 = -0.5\,E_h$ for hydrogen — no regression from the parity-aware padding refactor.

## 5. Balmer series from the log-radial ℓ=0 eigenvalues

For a photon of energy $\omega$ in Hartree, the vacuum wavelength is $\lambda[\text{nm}] = hc/\omega = 45.5633/\omega$ (where $hc = 1239.842\,\text{eV}\cdot\text{nm}$ and $1\,E_h = 27.2114\,\text{eV}$). The Balmer lines $n = 3, 4, 5 \to n = 2$ have

$$
\omega_{n \to 2} \;=\; E_n - E_2 \;=\; \frac{1}{8} - \frac{1}{2 n^2}. \tag{8}
$$

**Table 1.** Balmer-series line positions computed from `solve_bound_states(grid, Z=1.0, K=6, \ell=0)` on a log-radial grid with $N = 800$, $r_\text{min} = 10^{-3}$, $r_\text{max} = 120$, compared against the theoretical infinite-nuclear-mass Rydberg value $\lambda^{(\infty)}_{\rm theory} = hc/\omega_{n\to 2}$.

| Transition | $\omega$ (Ha, theory) | $\lambda^{(\infty)}_{\rm theory}$ (nm) | $\lambda_{\rm computed}$ (nm) | $|\Delta\lambda|$ (nm) | NIST $\lambda$ (air, nm) |
|------------|-----------------------|----------------------------------------|-------------------------------|-----------------------|--------------------------|
| H-α ($3 \to 2$) | $5/72 = 0.069444$  | $656.113$ | $656.113$ | $< 0.001$ | $656.28$ |
| H-β ($4 \to 2$) | $3/32 = 0.093750$  | $486.009$ | $486.009$ | $< 0.001$ | $486.13$ |
| H-γ ($5 \to 2$) | $21/200 = 0.105$   | $433.937$ | $433.937$ | $< 0.001$ | $434.05$ |

The Phase 6 exit criterion from `README.md` §5 — Balmer-α recovered to $< 0.1\,\text{nm}$ on an $N = 800$ log-radial grid — is met, and by more than two orders of magnitude. The residual ~0.2 nm offset vs NIST is the unmodeled reduced-mass correction $m_p/(m_p + m_e) \approx 1 - 1/1836$ relating the infinite-mass and physical Rydberg constants — a different shift than our numerical discretisation error, and outside the scope of a non-relativistic infinite-nuclear-mass solver.

## 6. Radial dipole and the Wigner–Eckart split

For single-nucleus problems the full E1 matrix element factorises into a radial integral and an angular Gaunt coefficient:

$$
\langle \psi_{n'\ell'm'} | \hat r_\alpha | \psi_{n\ell m}\rangle
\;=\; \langle u_{n'\ell'} | r | u_{n\ell}\rangle_r \cdot \langle Y_{\ell'm'} | \hat r_\alpha | Y_{\ell m}\rangle, \tag{9}
$$

with

$$
\langle u_{n'\ell'} | r | u_{n\ell}\rangle_r \;=\; \int_0^\infty u_{n'\ell'}(r)\,r\,u_{n\ell}(r)\,dr \tag{10}
$$

evaluated on the log grid as `gato.physics.radial_hydrogen.radial_dipole(u_f, u_i, grid)`. The angular factor is zero unless $\Delta\ell = \pm 1$ and $\Delta m \in \{-1, 0, +1\}$ — the E1 selection rules emerge from the angular integral, not from any postulate.

### 6.1 Cross-check: 1s → 2p on two independent stacks

For hydrogen the radial integral evaluates analytically to

$$
\langle u_{2p} | r | u_{1s}\rangle \;=\; \frac{256}{81\sqrt 6} \;\approx\; 1.29027\,a_0, \tag{11}
$$

and the angular Gaunt coefficient $\langle Y_{10} | \cos\theta | Y_{00}\rangle = 1/\sqrt 3$. The product,

$$
\langle \psi_{2p_z} | z | \psi_{1s}\rangle \;=\; \frac{256}{81\sqrt 6} \cdot \frac{1}{\sqrt 3} \;=\; \frac{256}{243\sqrt 2} \;=\; \frac{128\sqrt 2}{243}, \tag{12}
$$

recovers eq. (2). On the log-radial grid (`tests/test_radial_ell.py::test_radial_dipole_1s_to_2p_matches_analytic`) with $N = 800$, $r_\text{max} = 80\,a_0$, the computed radial integral agrees with eq. (11) to $10^{-3}\,a_0$, and the product with the analytic Gaunt factor matches the 3D Cartesian calculation. The two stacks — Cartesian `transition_dipole` with analytic orbitals, and log-radial `radial_dipole` with solver-produced $u_{1s}$, $u_{2p}$ — deliver the same number to three decimals.

## 7. Deferred work

- **Multi-electron excited states.** `spectra.excited_states_lanczos` on the Phase 3 Fock or Kohn–Sham Hamiltonian, for He $1s^2 \to 1s2p$ and heavier closed-shell atoms. Requires wiring the Phase 1 Lanczos driver to the Phase 3 SCF solution.
- **Angular Gaunt helper.** An explicit $\langle Y_{\ell'm'} | \hat r_\alpha | Y_{\ell m}\rangle$ utility so that solver-produced $(n, \ell)$ states on the log-radial grid feed directly into Cartesian oscillator strengths, enabling the $\ge 10^8$ forbidden-vs-allowed suppression demonstration (the headline selection-rule check of the Phase 6 exit criterion).
- **Absorption cross-section.** Lorentzian natural linewidth from eq. (5) plus Gaussian Doppler broadening at a user-supplied temperature, returning $\sigma(\omega)$ as a stick or smoothed curve, plus the `physics/atomic_spectra.py` end-to-end driver that turns a closed-shell atom into a NIST-comparable plot.

## 8. Reproducibility

```bash
uv run pytest tests/test_spectra.py tests/test_radial_ell.py -v    # ~20 s on CPU
```

All Phase 6 tests pass in double precision on a single CPU core. Ten new tests total, all green; full GATO suite 95 passed, 2 skipped (the GPU-gated Phase 3 Be/Ne SCF stubs).

## References

- Bethe, H. A. & Salpeter, E. E. (1957), *Quantum Mechanics of One- and Two-Electron Atoms*, Academic Press — §61–63 derive the closed-form hydrogen dipole matrix elements used throughout this note.
- Cowan, R. D. (1981), *The Theory of Atomic Structure and Spectra*, University of California Press — chapter 14 on oscillator strengths and the Wigner–Eckart split used in §6.
- NIST Atomic Spectra Database, <https://physics.nist.gov/asd> — Balmer-series reference wavelengths (air) used in Table 1.
- Griffiths, D. J. & Schroeter, D. F. (2018), *Introduction to Quantum Mechanics*, 3rd ed., Cambridge — §6 on time-dependent perturbation theory and oscillator strengths.
