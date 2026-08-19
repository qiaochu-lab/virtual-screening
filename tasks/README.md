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

Before quoting any number from these documents, read
[`../LIMITATIONS.md`](../LIMITATIONS.md) — it is ordered by how much each caveat
moves the headline claims, and the first two affect the main result directly.

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

## Where the code is

Each document ends with a **Code** table linking every claim in it to the script
that produced it. The scripts live next to what they operate on rather than in
one bucket:

| Directory | Contents |
|---|---|
| [`../t1/`](../t1/) | standard-benchmark launches, training-similarity ablation |
| [`../t3/build/`](../t3/build/) | time split, difficulty layers, eval-set construction |
| [`../t3/structure/`](../t3/structure/) | PDB metadata, chain mapping, pocket extraction at four thresholds |
| [`../t3/runners/`](../t3/runners/) | per-model adapters, and the patches each repo needed |
| [`../t3/analysis/`](../t3/analysis/) | main table, class breakdown, robustness controls |
| [`../physics/`](../physics/) | 🔬 FEP benchmark, Boltz-2 affinity, physics comparisons |
| [`../eval/`](../eval/) | metric implementations and their tests |
| [`../env/`](../env/) | per-model environment construction |

Every script carries a docstring saying what the check is for and **how to read
its output**, including what a null result would mean — written before the
output existed.
