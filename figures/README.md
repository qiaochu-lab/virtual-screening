# Figures

Regenerate all three with [`make_figures.py`](make_figures.py). It needs
matplotlib; on our machine only the `dock` environment has it.

```bash
/data/yicheng/xqc/envs/dock/bin/python make_figures.py
```

Each figure is written as both PNG (160 dpi) and PDF.

| File | What it shows | Source data |
|---|---|---|
| `fig1_t3_decay` | AUROC across the four time-split layers, one line per model, coloured by training set | [`../results/T3_main.csv`](../results/T3_main.csv) |
| `fig2_t1_heatmap` | EF at 1% for every model on the three standard benchmarks | [`../results/T1_main.csv`](../results/T1_main.csv) |
| `fig3_t6_physics` | Retrieval baseline vs physics rerank vs rank fusion, for both physics methods | [`../results/T6_rerank3.csv`](../results/T6_rerank3.csv), [`../results/T6_dock.csv`](../results/T6_dock.csv) |

**Reading fig 1 and fig 2 together.** Both are coloured by training set rather
than by architecture, because that is what separates the tiers: the four models
trained on PocketAffDB occupy the top four positions on DUD-E and the top four
curves at every layer of T3, and the three trained on the DrugCLIP set sit below
them — across substantial architectural differences. See
[`../LIMITATIONS.md`](../LIMITATIONS.md) on why this makes per-architecture
attribution unsafe.

**fig 3 excludes docking targets with incomplete ligand coverage** (three timed
out at 90 minutes and hold a non-random subset of ligands). Error bars are
standard error across targets; the n differs between panels because the two runs
used different shortlist depths and different layers.
