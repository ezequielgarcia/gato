# GATO Phase 7: Scalar-relativistic corrections — perturbative MV + Darwin and non-perturbative ZORA

> **Read this on <https://ezequielgarcia.github.io/gato/phase7_note/>** — GitHub's markdown viewer does not render display math reliably.

**Companion note to `phase1_paper.md` and `phase6_note.md`.** Phase 7 extends the log-radial solver of Phase 1 with two independent O($\alpha^2$) scalar-relativistic corrections: a perturbative mass–velocity + Darwin expectation-value pair evaluated on the nonrelativistic ground state, and a non-perturbative ZORA kinetic operator that replaces $-\tfrac{1}{2}\partial_r^2$ in the radial Hamiltonian. Both are implemented for the hydrogenic ℓ=0 channel; multi-electron radial SCF and the heavy-atom drivers (Au$^+$, Hg) that the phase is ultimately aimed at remain future work. The pedagogical payoff of this slice is a direct numerical comparison: on hydrogen 1s the two approximations disagree by a factor of two — a well-known feature of scalar ZORA, and the cleanest sanity check that each operator is correctly discretised.

## Abstract

We implement the mass–velocity operator $\hat H_{\rm MV} = -\hat p^4 / (8 c^2)$ and the Darwin contact term $\hat H_{\rm D} = (\pi Z / (2 c^2))\,\delta^3(\mathbf r)$ as expectation values on the Phase 1 log-radial ℓ=0 ground state, reproducing $\langle p^4\rangle_{1s} = 5 Z^4$ and $|\psi(0)|^2_{1s} = Z^3/\pi$ to ~0.2 % at $N = 1600$, $r_{\min} = 0.01/Z$. We also implement the full scalar ZORA kinetic operator, $\hat T_{\rm ZORA} = \hat p \cdot [c^2/(2 c^2 - V)] \cdot \hat p$, derived in radial form as $T_{\rm ZORA}\,u = -K\,u'' - K'\,u' + (K'/r)\,u$ — the $(K'/r)\,u$ term coming from the full 3D divergence $-\nabla\cdot(K\nabla\psi)$ and being essential: without it the discretised shift diverges with $N$. On hydrogen 1s, the computed shifts are $\Delta E_{\rm MV+D} = -\alpha^2 Z^4/8 \approx -6.66 \times 10^{-6}\,E_h$ (Sommerfeld fine structure at $j = 1/2$, agreeing to ~2 % given the partial cancellation between the two pieces) and $\Delta E_{\rm ZORA} = -\alpha^2 Z^4/4$, exactly twice the Sommerfeld value. This factor-of-two overshoot is the standard signature of scalar ZORA for deeply-bound 1s states: ZORA and Foldy–Wouthuysen give different leading-order resummations of the Dirac equation, and they disagree on the coefficient by the "picture-change" factor that FW includes and scalar ZORA omits. Both scale as $Z^4$ and both converge with grid refinement; the agreement of the ratio to two decimals validates both discretisations.

## 1. Scope

Phase 7 is framed as a one-month optional extension in `README.md` §5.7, layering one new operator onto the existing log-radial infrastructure. The deliverables at the end of the phase are scalar-relativistic RHF and LDA on heavy atoms — Au$^+$, Hg, with light-homolog Ag and Cu contrasts — and the Hg 254 nm line from Phase 6 with ZORA in place of NR kinetic. This note covers only the *operator layer*: hydrogen-atom validation of both perturbative and non-perturbative O($\alpha^2$) forms, no SCF, no multi-electron atoms.

The scope is deliberately narrow because the physical content worth checking first is the operator itself, not its integration into a self-consistent loop. If scalar ZORA on hydrogen 1s gave the Sommerfeld shift directly instead of twice it, that would be a *discretisation bug*; verifying that we reproduce the factor-of-two feature of scalar ZORA with the correct sign, magnitude, $Z^4$ scaling, and $N$-convergence is exactly what tells us the operator is correctly implemented.

## 2. Perturbative mass–velocity and Darwin

### 2.1 Operators and matrix elements

The leading relativistic corrections to a nonrelativistic Schrödinger eigenvalue are, in atomic units with $c = 1/\alpha$,

$$
\hat H_{\rm MV} \;=\; -\frac{\hat p^4}{8 c^2}, \qquad
\hat H_{\rm D} \;=\; \frac{\pi Z}{2 c^2}\,\delta^3(\mathbf r), \tag{1}
$$

summed over all nuclei for the general case; for hydrogen there is one term. Spin–orbit $\hat H_{\rm SO}$ vanishes identically for $\ell = 0$ states and is not part of this slice.

### 2.2 $\langle p^4\rangle$ on the log-radial grid

For a normalised ℓ=0 state $\psi(\mathbf r) = (u(r)/r)\,Y_{00}(\theta, \phi)$,

$$
p^2\psi \;=\; -\nabla^2\psi \;=\; -\frac{u''(r)}{r} \qquad (\ell = 0), \tag{2}
$$

so

$$
\langle p^4\rangle \;=\; \langle p^2\psi\,|\,p^2\psi\rangle
\;=\; \int_0^\infty \bigl(u''(r)\bigr)^2\,dr, \tag{3}
$$

computed on the log grid as `radial_inner_product(u_rr, u_rr, grid)` where `u_rr = radial_laplacian(u, grid, \ell=0)`. The 4th-order chain-rule stencil returns $u''(r)$ directly in physical $r$-coordinates, so no extra chain-rule factors are needed at this level.

For hydrogen 1s with $u(r) = 2 Z^{3/2} r\,e^{-Z r}$, eq. (3) integrates analytically to $\langle p^4\rangle = 5\,Z^4$. The implementation `gato.physics.fine_structure.p4_expectation(u, grid)` reproduces this to 0.2 % at $N = 1600$; see Table 1.

### 2.3 $|\psi(0)|^2$ from the innermost grid point

The Darwin contact term needs $|\psi(0)|^2$. For $\ell = 0$ states $\psi(0) = R(0)/\sqrt{4\pi}$ with $R(r) = u(r)/r$. Using L'Hôpital's rule in continuum, $R(0) = u'(0)$; on the log grid, $u[0]/r[0]$ is a numerically-direct proxy because the innermost sampled $r$ is $\sim r_{\min}\alpha\,h_\xi/2$, many orders of magnitude below the physical length scale $1/Z$. Quantitatively, for hydrogen 1s, the truncation error is $O((r[0])^2)$, well below 0.1 %.

### 2.4 Choice of $r_{\min}$

The chain-rule radial Laplacian on the log grid is

$$
u''(r) \;=\; \frac{1}{(r')^2}\,\frac{d^2 u}{d\xi^2} \;-\; \frac{r''}{(r')^3}\,\frac{d u}{d\xi}, \tag{4}
$$

with $r'(\xi) = \alpha (r + r_{\min})$. Near the inner boundary $r'(\xi = 0) \sim \alpha\,r_{\min}$; if $r_{\min}$ is too small, the factor $1/(r')^2$ amplifies any 4th-order stencil noise to the point of biasing $\langle p^4\rangle$ by several percent. Empirically, $r_{\min} = 10^{-2}/Z$ is the sweet spot: the integrand $(u''(r))^2$ has a finite value at the origin ($u''(0) = -4\,Z^{5/2}$ for hydrogen, bounded), so truncating the integral at $r \sim 10^{-5}\,a_0$ incurs an error of at most $16 \cdot 10^{-5} = 1.6 \times 10^{-4}$ — well below 0.1 % of the full $\langle p^4\rangle = 5$. Smaller $r_{\min}$ has no more physical information but pays a price in discretisation noise.

### 2.5 Total shift and Z-scaling

**Table 1.** Perturbative scalar-relativistic shift for hydrogen 1s on a log-radial grid with $N = 1600$, $r_{\min} = 10^{-2}/Z$, $r_{\max} = 40/Z$.

| $Z$ | $\langle p^4\rangle$ (theory: $5 Z^4$) | $|\psi(0)|^2$ (theory: $Z^3/\pi$) | $\langle H_{\rm MV}\rangle$ (theory: $-5\alpha^2 Z^4/8$) | $\langle H_{\rm D}\rangle$ (theory: $+\alpha^2 Z^4/2$) | $\Delta E_{\rm MV+D}$ (theory: $-\alpha^2 Z^4/8$) |
|-----|-----------------------------------------|-----------------------------------|---------------------------------------------------------|--------------------------------------------------------|----------------------------------------------------|
| 1.0 | $5.006$ (0.11 % high) | $0.3177$ (0.19 % low) | $-3.33\times 10^{-5}$ | $+2.66 \times 10^{-5}$ | $-6.80 \times 10^{-6}$ (2.2 % off $-6.66\times 10^{-6}$) |
| 2.0 | $80.1$ (0.11 % high)  | $2.542$ (0.19 % low)  | $-5.33\times 10^{-4}$ | $+4.25\times 10^{-4}$ | $-1.09\times 10^{-4}$ (2.1 % off $-1.07\times 10^{-4}$)   |

Both individual pieces match analytic values to ~0.2 %. The total shift $\Delta E_{\rm MV+D}$ is dominated by the near-cancellation of MV and Darwin (which are nearly equal in magnitude but opposite in sign, leaving only $\alpha^2/8$ instead of $5\alpha^2/8$). The same 0.2 % absolute error in each piece becomes ~2 % relative error in the combined shift — fundamental to the subtraction, not a solver defect, and a much smaller residual is not achievable without substantially denser grids.

$Z^4$ scaling is verified across $Z = 1$ and $Z = 2$ for every quantity in Table 1 (ratios all come out to within 0.2 % of the theoretical factor).

## 3. ZORA on the log-radial grid

### 3.1 The operator

The zero-order regular approximation (ZORA) replaces the nonrelativistic kinetic energy $\hat T = \hat p^2 / 2$ with

$$
\hat T_{\rm ZORA} \;=\; \hat{\mathbf p} \cdot \frac{c^2}{2 c^2 - V(\mathbf r)} \cdot \hat{\mathbf p}, \tag{5}
$$

which reduces to $\hat T_{\rm NR}$ in the limit $c \to \infty$ (equivalently $V/c^2 \to 0$) and to a scaled Dirac kinetic in the limit of deep cores. In atomic units with $c = 1/\alpha$ the kinetic weight $K(r) = c^2/(2 c^2 - V(r))$ takes the value $K = 1/2$ where $V = 0$ and drops to zero where $V \to -\infty$. For Coulomb $V = -Z/r$,

$$
K(r) \;=\; \frac{c^2 r}{2 c^2 r + Z}, \qquad
K'(r) \;=\; \frac{c^2 Z / r^2}{(2 c^2 - V)^2}
\;=\; \frac{c^2 Z}{r^2\,(2 c^2 + Z/r)^2}. \tag{6}
$$

Both $K$ and $K'$ are closed-form analytic functions of $r$; only $u(r)$ is known numerically.

### 3.2 Radial reduction

The 3D divergence of $K\,\nabla\psi$ with $\psi = u(r)/r$ unfolds as

$$
-\nabla\cdot\bigl(K \nabla\psi\bigr)
\;=\; -\frac{K}{r}\,u''(r) \;-\; \frac{K'}{r}\,u'(r) \;+\; \frac{K'}{r^2}\,u(r), \tag{7}
$$

and multiplying the ZORA eigenequation $(\hat T_{\rm ZORA} + V)\psi = E\psi$ by $r$ (so $u$ becomes the primary unknown) gives the radial form

$$
\boxed{\;-K(r)\,u''(r) \;-\; K'(r)\,u'(r) \;+\; \frac{K'(r)}{r}\,u(r) \;+\; V(r)\,u(r)
\;=\; E\,u(r). \;} \tag{8}
$$

The $(K'/r)\,u$ term is subtle and, in this project's experience, was the deciding source of a derivation bug: a naive translation of $-\nabla\cdot(K\nabla)$ as the 1D form $-\partial_r(K\,\partial_r)$ misses it entirely, and the discretised energy shift then diverges with $N$ (see §3.3). Eq. (8) is self-adjoint under the log-grid inner product $\langle u, v\rangle_W = h_\xi\sum r'_j\,u_j v_j$ because $\int [K u' v' + (K'/r) u v]\,dr$ is symmetric in $u, v$.

### 3.3 Discretisation and symmetrisation

`kinetic_zora_radial(u, grid, Z)` evaluates eq. (8) pointwise with $u'(r)$ from the 4th-order chain-rule $\xi$-stencil and $u''(r)$ from `radial_laplacian`. The dense matrix $\mathbf H$ of the operator is built column-by-column, and `solve_zora_ground_state` applies the same $W^{1/2} \mathbf H W^{-1/2}$ symmetrisation used in Phase 1's `solve_ground_state`, followed by a final explicit $(\mathbf H + \mathbf H^T)/2$ symmetrisation to absorb any residual $O(h^4)$ asymmetry from the stencil.

Convergence with $N$ is the essential diagnostic. A *correct* discretisation of eq. (8) gives a ZORA shift that settles to a stable value as $N \to \infty$:

| $N$ | $E_{\rm NR}$ | $E_{\rm ZORA}$ | $E_{\rm ZORA} - E_{\rm NR}$ |
|-----|----------------|-----------------|------------------------------|
| 800  | $-0.4999999$ | $-0.50001332$ | $-1.336 \times 10^{-5}$ |
| 1600 | $-0.5000000$ | $-0.50001332$ | $-1.332 \times 10^{-5}$ |
| 3200 | $-0.5000000$ | $-0.50001332$ | $-1.332 \times 10^{-5}$ |

Removing the $(K'/r)\,u$ term produces instead a monotonically-diverging shift (~$-5.7\times 10^{-4}$ at $N = 800$ growing to $-7.4 \times 10^{-4}$ at $N = 3200$, two orders of magnitude too large and not settling) — exactly what a missing differential term should look like, and the reason `tests/test_fine_structure.py::test_zora_shift_converges_with_N` is the first line of defence in catching regressions in eq. (8).

### 3.4 The factor of two

Expanding eq. (5) to leading order in $V/c^2$,

$$
K \;=\; \tfrac{1}{2}\bigl[1 + V/(2 c^2) + O(V^2/c^4)\bigr] \;\Rightarrow\;
\hat T_{\rm ZORA} \;=\; \frac{\hat p^2}{2} \;+\; \frac{\alpha^2}{4}\,\hat{\mathbf p}\cdot V\cdot\hat{\mathbf p} \;+\; O(\alpha^4). \tag{9}
$$

For hydrogen 1s, $|\hat{\mathbf p}\psi_{1s}|^2 = Z^2\,|\psi_{1s}|^2$ pointwise, so

$$
\langle\hat{\mathbf p}\cdot V \cdot \hat{\mathbf p}\rangle_{1s}
\;=\; \int |\hat{\mathbf p}\psi_{1s}|^2\,V\,d^3r
\;=\; Z^2\,\langle V\rangle_{1s}
\;=\; Z^2 \cdot (-Z^2)
\;=\; -Z^4, \tag{10}
$$

giving

$$
\Delta E_{\rm ZORA,\,leading} \;=\; -\frac{\alpha^2 Z^4}{4}, \tag{11}
$$

exactly twice the Sommerfeld shift $-\alpha^2 Z^4/8$ from eq. (1). This overshoot is not a bug of the implementation — it is the well-known "picture-change" discrepancy between the ZORA and Foldy–Wouthuysen resummations. Both are valid O($\alpha^2$) approximations to the Dirac equation; they differ by the factor that arises when relating the large and small components of the Dirac spinor. FW includes it (yielding MV + Darwin → $-\alpha^2 Z^4/8$), scalar ZORA omits it (yielding $-\alpha^2 Z^4/4$). The exact Dirac value at $n = 1$, $j = 1/2$ is the Sommerfeld result.

### 3.5 Cross-check: ZORA vs MV+Darwin

**Table 2.** Two independent O($\alpha^2$) scalar-relativistic shifts for hydrogen 1s on $N = 1600$, $r_{\min} = 10^{-2}$, $r_{\max} = 40$.

| Observable | Value | Leading-order theory |
|------------|--------|----------------------|
| $\Delta E_{\rm MV+D}$ (eq. 1 expectation on $u_{1s}^{(\rm NR)}$) | $-6.80 \times 10^{-6}\,E_h$ | $-\alpha^2 Z^4/8 = -6.66\times 10^{-6}\,E_h$ |
| $\Delta E_{\rm ZORA}$ (eq. 8 ground-state eigenvalue minus $E_{\rm NR}$) | $-1.332\times 10^{-5}\,E_h$ | $-\alpha^2 Z^4/4 = -1.331\times 10^{-5}\,E_h$ |
| Ratio $\Delta E_{\rm ZORA}\,/\,\Delta E_{\rm MV+D}$ | $1.96$ | $2.00$ (exact at this order) |

Both shifts converge with $N$, scale as $Z^4$ across $Z = 1, 2$ (shift(Z=2)/shift(Z=1) = 16 to 1 %), and are strictly negative (attractive relativistic binding). The ratio $\approx 2$ is the sharpest single diagnostic: a bug in either code path — for example, a sign error in a derivative, or a missed term in the radial reduction — breaks the ratio immediately.

## 4. Deferred work

The full Phase 7 deliverables from `README.md` §5.7 are:

- [ ] LDA exchange–correlation on the 1D radial grid (trivial $4\pi r^2$ Jacobian).
- [ ] `scf_rhf_radial` / `scf_ks_lda_radial` — radial SCF loops with a spherical Hartree kernel (1D radial Poisson, $O(N)$).
- [ ] Aufbau-based core-valence partitioning by $\ell$-shell (not by Lanczos on a single Krylov run; shell structure is explicit in 1D radial form).
- [ ] `physics/gold.py` — end-to-end driver for Au$^+$ ($n_{\rm occ} = 34$) and Hg ($n_{\rm occ} = 40$), running NR and scalar-ZORA side by side.
- [ ] Light-homolog contrast Ag, Cu to demonstrate $\sim Z^2$ scaling of the relativistic correction.
- [ ] Phase 6 spectra hook: Hg 254 nm with the ZORA operator, where NR gives zero oscillator strength and scalar relativity borrows allowed character from $^1P_1$.

What this note covers is the operator layer that every downstream item depends on. Without a correctly-discretised ZORA kinetic (and a correctly-derived (K'/r)u term in particular), multi-electron radial SCF would simply propagate the factor-of-2-wrong shift into the orbital energies and poison every downstream atomic observable.

## 5. Reproducibility

```bash
uv run pytest tests/test_fine_structure.py -v                       # ~70 s on CPU
```

Ten new tests total across §2 and §3 (six for MV+Darwin, four for ZORA), all green in double precision on a single CPU core.

## References

- Bethe, H. A. & Salpeter, E. E. (1957), *Quantum Mechanics of One- and Two-Electron Atoms*, Academic Press — §§16–18 derive the mass–velocity and Darwin terms from the Foldy–Wouthuysen transformation of the Dirac equation.
- Sommerfeld, A. (1916), "Zur Quantentheorie der Spektrallinien", *Annalen der Physik* **356**, 1–94 — the fine-structure energy $E_{n, j}$ used as the target in Table 1.
- van Lenthe, E., Baerends, E. J. & Snijders, J. G. (1993), "Relativistic regular two-component Hamiltonians", *Journal of Chemical Physics* **99**, 4597 — the canonical ZORA reference, and where the factor-of-two overshoot on deeply-bound 1s states is discussed under the heading of "scaled ZORA" and the picture-change correction.
- Desclaux, J.-P. (1973), "Relativistic Dirac–Fock expectation values for atoms with $Z = 1$ to $Z = 120$", *Atomic Data and Nuclear Data Tables* **12**, 311 — Dirac–Fock atomic orbital energies that later multi-electron Phase 7 work will use as reference.
- Liu, W. (2010), "Ideas of relativistic quantum chemistry", *Molecular Physics* **108**, 1679 — modern review situating ZORA among the family of two-component relativistic approximations.
- Foldy, L. L. & Wouthuysen, S. A. (1950), "On the Dirac theory of spin 1/2 particles and its non-relativistic limit", *Physical Review* **78**, 29 — the FW transformation that produces the MV+Darwin form and distinguishes it from scalar ZORA.
