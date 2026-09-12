# Figures

Regenerate all three with [`make_figures.py`](make_figures.py). It needs
matplotlib; on our machine only the `dock` environment has it.

```bash
/data/work/envs/dock/bin/python make_figures.py
```

Each figure is written as both PNG (160 dpi) and PDF.

| File | What it shows | Source data |
|---|---|---|
| `fig1_t3_decay` | AUROC across the four time-split layers, one line per model, coloured by training set | [`../results/T3_main.csv`](../results/T3_main.csv) |
| `fig2_t1_heatmap` | EF at 1% for every model on the three standard benchmarks | [`../results/T1_main.csv`](../results/T1_main.csv) |
| `fig3_t6_physics` | Retrieval baseline vs physics rerank vs rank fusion, for both physics methods | [`../results/T6_rerank3.csv`](../results/T6_rerank3.csv), [`../results/T6_dock.csv`](../results/T6_dock.csv) |

**fig 1–3 were re-rendered on 2026-09-12**, after the HypSeek screening rows
switched from `_rk` to the official `_vs` (see [`../MODELS.md`](../MODELS.md)).
`make_figures.py` was updated at the same time so the new row is named
`HypSeek` and stays in the PocketAffDB colour group. The point both figures
exist to make is unchanged: `_vs` remains in that group at every layer, and the
four PocketAffDB models still hold the top four places on DUD-E and at all four
T3 layers.

⚠️ fig 3 is driven by the T6 tables, which this change did not touch — its
bytes differ only because the whole set was re-rendered under a newer
matplotlib. fig 4 comes from a separate script (`fig4_actives.py`) and was not
regenerated.

**Reading fig 1 and fig 2 together.** Both are coloured by training set rather
than by architecture, because that is what separates the tiers: the four models
trained on PocketAffDB occupy the top four positions on DUD-E and the top four
curves at every layer of T3, and the three trained on the DrugCLIP set sit below
them — across substantial architectural differences. See
[`../LIMITATIONS.md`](../LIMITATIONS.md) on why this makes per-architecture
attribution unsafe.

**fig 3 keeps docking targets with incomplete coverage** and labels how many
(four timed out at 90 minutes, coverage 36–90%). They are kept because both arms
are scored on the same ligand subset, so the comparison the figure makes stays
valid even though the absolute AUROC is biased. `score_dock.py` prints the
complete-only summary as well; it points the same way with n = 5. Error bars are
standard error across targets, and n differs between panels because the two runs
used different shortlist depths and different layers.

| `fig4_actives_per_target` | How many actives each target has, and what that does to EF's resolution | [`../results/T3_targets.csv`](../results/T3_targets.csv) |

**fig 4 explains a limitation rather than a result.** EF@1%'s step size is
about 100/A for a target with A actives, so at the ≥10 floor one extra hit moves
EF by 8.5 — inside the range of the layer means themselves. The right panel puts
that curve against the shaded band of layer means. The gradient analysis that
follows from it is in
[`../results/T3_actives_gradient.csv`](../results/T3_actives_gradient.csv);
rebuild the figure with [`fig4_actives.py`](fig4_actives.py).
