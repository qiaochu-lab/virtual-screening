# Tasks T1–T6

One document per task. Each states the question, the data, how it was run, what
came out, and what the result does **not** support.

| Task | Question | Status |
|---|---|---|
| [T1 — Enrichment](T1-enrichment.md) | do published enrichment numbers reproduce? | 3 of 9 models |
| [T2 — Affinity ranking](T2-affinity-ranking.md) 🔬 | can they rank binding strength? | ✅ answered |
| [T3 — Time-split](T3-time-split.md) | do they generalise past the training cutoff? | ✅ main result |
| [T4 — Target fishing](T4-target-fishing.md) | molecule → target | not started |
| [T5 — Structure robustness](T5-structure-robustness.md) 🔬 | are the conclusions stable? | ✅ two controls |
| [T6 — Physics complementarity](T6-physics.md) 🔬 | can physics fill the ranking gap? | running |

🔬 marks a task with a **"where physics fits"** section.

## For physics collaborators

Read in this order:

1. **[T2](T2-affinity-ranking.md)** — the problem. Retrieval models produce
   cosine similarity, not free energy: no ΔG, no RMSE in kcal/mol, no ΔΔG. Every
   thermodynamic quantity is currently missing from this benchmark.
2. **[T6](T6-physics.md)** — the four concrete entry points, what data already
   exists for each, and the five known traps.
3. **[T5](T5-structure-robustness.md)** — pocket definitions are prepared at
   4/5/6/8 Å with the sensitivity quantified; any structure-based method needs
   this.

The interface is deliberately minimal: a physics method contributes one float
array per target (plus the label array), and [`../eval/`](../eval/) computes
every metric from that. Nothing in the metric layer is model-specific.

## Scripts

[`scripts/`](scripts/) holds the analysis code behind the numbers in these
documents — one script per claim, each with a docstring explaining what the
check is for and how to read its output. Dataset-construction code lives in
[`../t3/`](../t3/).
