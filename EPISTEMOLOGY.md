# Epistemology notes

What *kind* of thing is differentiable physics, philosophically?

## Kuhn

Thomas Kuhn (*The Structure of Scientific Revolutions*, 1962) distinguished
**normal science** — puzzle-solving inside an accepted paradigm — from
**paradigm shifts**, triggered by persistent anomalies the reigning framework
can't absorb. His picture: theory proceeds by long calm stretches punctuated
by crises, after which the community adopts a new paradigm that is
*incommensurable* with the old one (different questions count as important,
different answers count as valid).

Applied to GATO / differentiable physics: **this is not a paradigm shift**.
The underlying physics — Schrödinger's equation, Hamilton's principle, the
variational calculus connecting them — was settled by the 1920s, and Dirac
pointed out the formal analogy between classical action and quantum
amplitude in 1933. What's new is *technique*: JAX + autodiff lets us execute
variational principles numerically in cases that were previously pen-and-paper
or bespoke Fortran. Kuhn carefully separated tools from paradigm-constitutive
commitments and would file this under the former.

The sharper Kuhnian point: tools quietly reshape *which problems seem
interesting*. Once "minimize any energy functional over any parameterized
ansatz" is cheap, the questions drift — less "what is the ground state of He"
(solved) and more "what does pseudopotential design space look like under
gradient flow" (newly askable). Whether any of those new exhibits eventually
surfaces a genuine anomaly is the open question. For now: better microscope,
not new biology.

### Do we have enough for a crisis?

No, nowhere close. Kuhn's crisis precondition is *persistent, recognized
anomalies* the paradigm has tried and failed to absorb. Computational
electronic structure has the opposite problem — it works almost too well:
RHF, DFT, CCSD(T), FermiNet form a ladder that agrees with experiment at
increasing cost, and when it disagrees, everyone knows which approximation
to blame. That is textbook normal science.

The real candidate-anomalies in physics today — dark matter/energy, the
measurement problem, hierarchy/fine-tuning, quantum gravity — live several
floors above where a grid-based Schrödinger solver operates. Neural ansätze
+ autodiff are a better wrench, not evidence that the engine is broken.

## Latour

Bruno Latour (1947–2022). Where Kuhn asked *how do scientific theories
change*, Latour asked the more deflationary *how do scientific facts get
made in the first place*. *Laboratory Life* (1979, with Woolgar) was an
ethnography of the Salk Institute: Latour sat in a biology lab as an
anthropologist and traced how messy gel photographs, arguments, grants, and
citations gradually hardened into the accepted "fact" that TRH is
Pyro-Glu-His-Pro-NH₂.

His framework is **Actor-Network Theory**: a scientific result is not the
output of a lone mind discovering nature but a stable *network* of humans,
instruments, papers, funders, and objects reinforcing each other. A fact is
whatever a network has managed to make indispensable; pull out any node —
the machine, the lab, the consensus — and the fact dissolves.

## Why bring Latour in here

The interesting shift happening in ML-for-science isn't Kuhnian (no theory is
breaking) — it's Latourian. The *network* is changing:

- neural nets are new non-human actors in the lab,
- a GPU cluster functions as something between an instrument and a co-author,
- a trained weight file is cited like a reagent,
- what counts as *a result* expands from "a proof or measurement" to "a
  model that predicts well on held-out data."

That's the kind of quiet reshaping of scientific practice Latour was good at
noticing, and it sits orthogonally to the Kuhnian question of whether any
theory is in crisis.
