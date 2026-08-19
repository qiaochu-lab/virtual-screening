# t1/ — standard-benchmark runs and the training-similarity ablation

Launch scripts for DUD-E / LIT-PCBA / DEKOIS, plus the analysis of how much
performance depends on proximity to the training set. See
[`../tasks/T1-enrichment.md`](../tasks/T1-enrichment.md).

| File | What it does |
|---|---|
| `run_drugclip.sh` | DrugCLIP on DUD-E and LIT-PCBA |
| `run_bindclip.sh`, `run_bindclip_pcba.sh` | BindCLIP, both released weights |
| `run_dekois.sh` | DEKOIS 2.0 |
| `t1_sim.py` | EF vs sequence identity to the nearest training protein |
| `t1_sim3.py` | same, but as distance to the nearest *remaining* training protein — test proteins were removed from training, so the meaningful quantity is how close the closest survivor is |
| `build_train_union.py` | union of the evaluated models' training sets (there are only two distinct sets across seven pocket models) |
| `quantify_train_union.py` | how many L3/L4 targets are actually familiar to a model whose training set differs from the one used to assign layers |
| `check_pair_contamination.py` | pair-level check: how many (target, ligand) pairs in T3 already appear in training |

The last two exist because the time split alone does not guarantee novelty: a
2025 database record may describe a pair measured years earlier. Both scripts
quantify how much that inflates the control layers — and therefore how much the
L1→L4 decay could be overstated.

Bootstrapping in `t1_sim*.py` resamples **targets**, not molecules. Pooling
molecules across targets understates the variance and quietly changes what EF
means.
