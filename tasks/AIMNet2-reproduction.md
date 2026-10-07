# AIMNet2: Experiment Reproduction and Testing on Our T3 Data

This page documents two tasks: independently rerunning the main experiments with traceable inputs in the linked source repository, and applying its latest pipeline to our frozen T3 data. This is a supplementary analysis; the project's primary results should still be cited from the [canonical tables](../results/CANONICAL_TABLES.md).

The main findings are: **the FEP affinity-ranking signal was reproduced; CASF energy reranking did not improve the retrieval-selected candidates; most original T3 layer metrics were close, although fresh docking and some optimization endpoints differed. Our own T3 sample also showed no consistent affinity-ranking advantage.**

Agreement has two distinct meanings here:

1. **Reproducing the original experiments:** rerunning the original data, scientific code, weights, and parameters, then comparing against the published results.
2. **Testing on our data:** applying the same pipeline to a different T3 sample and observing similarly limited ranking signals. This is agreement in the overall trend; numerical equality between different samples is not expected.

## 1. Original Work and Reproduction Coverage

The source is [the upstream AIMNet2 scoring repository](https://github.com/xianyang123-bit/aimnet_score_pipelines), pinned to its latest commit [`e6a598bf80f5`](https://github.com/xianyang123-bit/aimnet_score_pipelines/tree/e6a598bf80f5e284e8a6515b507a2b3192546b62). Deleted inputs and earlier results were recovered from historical commit [`18072caf3d3d`](https://github.com/xianyang123-bit/aimnet_score_pipelines/tree/18072caf3d3db434f328f44ca3e5e39366382f5b), with Git blob identities verified during download.

The original repository constructs two scores from public AIMNet2 potential-energy models. The latest protocol uses fixed-receptor interaction energy, `Eint = Ecomplex − Epocket − Eligand`; the earlier composite score also includes desolvation and local ligand strain. Lower energies rank higher. This reconstructs publicly described scoring expressions and should not be presented as the official affinity-trained AIMNet2(Score) model.

| Experiment | Independently executed scope | Status |
| --- | --- | --- |
| KIN66 / PLA15 reference energies | 81 systems, two geometries, seven members or computational variants; 1,134 records | Complete |
| Latest FEP fixed-receptor Eint | 16 requested systems, 14 supported systems, 403 successful ligands; pockets prepared again and ligands optimized | Complete |
| CASF LigUnity retrieval | Embeddings regenerated; 24 supported pockets × 285 molecules, 6,840 score pairs | Complete |
| CASF revised energy reranking | 24 pockets, 480 ligands, 47,896 available poses; a separate 2,000-pose example | Complete |
| Latest original T3 pipeline | Pocket preparation and fresh docking rerun; separate energy rerun with published poses held fixed | Both complete: 41 systems / 403 ligands |
| Earlier original T3 composite | Old wB97M and revised protocols, each with 93 systems / 917 ligands, using the corresponding archived scientific code | Complete |
| Nonrandom smoke set | 250 molecules, 58 actives / 192 decoys; old and revised protocols independently calculated | Complete |
| Our T3 data | Frozen sample of 92 target-layer combinations / 920 active ligands | Complete: 67 systems / 547 ligands scored successfully |

The old and revised archived protocols each comprised 134 execution cases, totaling 268. Input and weight checks, molecule IDs, and energy-component identities were verified. The first pass with the revised model produced one nonfinite result; one retry with the same parameters returned a finite value. The old protocol had no scoring errors. Finite scores were not replaced based on their similarity to the reference results.

## 2. How the Original Results Agree

### FEP: An Affinity-Ranking Signal Was Reproduced

FEP is the dataset name in this pipeline. We calculated fixed-receptor interaction energies; we did not run free-energy perturbation simulations. The metric is `Spearman(−Eint, pAffinity)` within each system, followed by an equally weighted average across systems.

| Same 14 systems / 403 ligands | Published reference | Independent rerun |
| --- | --- | --- |
| Mean Spearman before optimization | 0.2713 | 0.2713 |
| Mean Spearman after optimization | 0.4992 | 0.5002 |

All 403 successful ligands matched individually, with identical labels and no missing or additional entries. The mean absolute difference in interaction energy after optimization was 0.0320 kcal/mol, with a maximum of 4.3828 kcal/mol. The signal means that model rankings show some agreement with experimental binding strength; it does not establish accurate Kd / Ki predictions or effectiveness in every system. See the [14-system comparison](../results/aimnet2-reproduction-20261007/fep_system_comparison.csv) and the [original FEP description](https://github.com/xianyang123-bit/aimnet_score_pipelines/blob/e6a598bf80f5e284e8a6515b507a2b3192546b62/work/FEP-fixed-receptor-eint-20260921/README.md).

On the same ligands, strictly paired with existing Boltz predictions, Boltz's mean correlation was 0.6391. The mean system-level difference, AIMNet minus Boltz, was −0.1389, with a system-bootstrap 95% interval of [−0.2914, 0.0079]. This sample does not establish general superiority or inferiority. Boltz inference was not rerun.

### CASF: Reranking Did Not Improve the Candidate Set

| Mean shortlist EF1 across 24 pockets | Published reference | Independent rerun |
| --- | --- | --- |
| LigUnity retrieval | 3.2024 | 3.2024 |
| Interaction before optimization | 1.6389 | 1.6389 |
| Interaction after optimization | 0.9722 | 0.9722 |
| Composite | 0.9722 | 0.9722 |

Each pocket has only 20 retrieval-selected candidates. The EF1 cutoff therefore selects the top-ranked molecule, normalized by the active fraction in that candidate set. This is not full-library screening EF1. The mean number of actives in the top 10 after interaction-energy optimization was 2.6667, compared with 2.7083 in the original report, so not every metric was identical. The old protocol's interaction / composite EF1 values of 0.7579 / 0.9722 were also reproduced.

The maximum difference between independently generated retrieval scores and the reference was 0.00114. The top 20 matched exactly for 23 pockets; one boundary candidate differed for the remaining pocket. Energy reranking used the original candidate set to preserve the comparison. After rescoring all 47,896 poses, the final pose selections for all 480 ligands matched the original report. Across 49,896 single-point energies, including the separate example, the mean absolute difference was 0.00238 kcal/mol.

The supported conclusion is that **energy reranking did not improve these retrieval-selected candidates**. This does not establish that retrieval itself performs poorly overall. See the [CASF comparison](../results/aimnet2-reproduction-20261007/casf_metrics_comparison.csv) and the [original CASF / composite description](https://github.com/xianyang123-bit/aimnet_score_pipelines/blob/e6a598bf80f5e284e8a6515b507a2b3192546b62/work/aimnet2025-benchmarks-20260906/README.md).

### Original T3: Similar Metric Trends, with Different End-to-End Conformations

Of the 96 systems requested by the latest original protocol, 41 were prepared successfully, 25 had unsupported chemistry, 20 had unrecovered source structures, 7 failed preparation, and 3 had empty original inputs. These coverage categories matched the original run. Heavy-atom coordinates, atom order, and net charges matched for all 41 prepared pockets. All coordinates matched for 39 pockets; the other two differed only in hydrogen coordinates.

| Latest Eint mean Spearman | Published reference | Published poses held fixed | Fresh docking from scratch |
| --- | --- | --- | --- |
| L1 | −0.1161 | −0.1161 | −0.1931 |
| L2 | 0.0759 | 0.0825 | 0.1124 |
| L3 | 0.0633 | 0.0579 | 0.1006 |
| L4 | 0.0647 | 0.0770 | 0.0493 |

With the published inputs held fixed, the mean absolute energy difference before optimization was only 0.00139 kcal/mol. After optimization, the mean difference was 0.05463, with a maximum of 8.08770. Fresh docking produced a median heavy-atom RMSD of 1.188 Å and a maximum of 30.956 Å; the mean energy difference after optimization relative to the original records was 8.346 kcal/mol. We therefore do not claim exact reproduction of end-to-end conformations or every energy value. See the [41-system comparison](../results/aimnet2-reproduction-20261007/latest_t3_per_target_comparison.csv) and the [original latest T3 description](https://github.com/xianyang123-bit/aimnet_score_pipelines/blob/e6a598bf80f5e284e8a6515b507a2b3192546b62/work/L1234-fixed-receptor-eint-20260921/README.md).

Both earlier T3 composite experiments were also independently rerun, each on the same 93 systems / 917 ligands:

| Layer | Revised reference composite ρ | Revised rerun ρ | Old reference composite ρ | Old rerun ρ |
| --- | --- | --- | --- | --- |
| L1 | −0.0877 | −0.0939 | −0.0165 | 0.0018 |
| L2 | 0.1018 | 0.1058 | 0.0441 | 0.0466 |
| L3 | 0.2174 | 0.2121 | 0.0698 | 0.0650 |
| L4 | 0.1169 | 0.1186 | 0.0177 | 0.0160 |

The earlier L3 result with the revised model does show some positive correlation; it would be inaccurate to say that all T3 experiments have no signal. Pocket chemistry, sample coverage, and pose selection changed between protocols, so these tables cannot isolate the effect of a single model or pipeline change. See the [186 system-version records](../results/aimnet2-reproduction-20261007/composite_t3_per_target_comparison.csv).

### Smoke and Quantum References: Independently Recomputed

| AUROC on the nonrandom 250-molecule smoke set | Published reference | Independent rerun |
| --- | --- | --- |
| Revised interaction before optimization | 0.5796 | 0.5796 |
| Revised interaction after optimization | 0.5422 | 0.5427 |
| Revised composite | 0.5216 | 0.5224 |
| Old interaction after optimization | 0.4543 | 0.4497 |
| Old composite | 0.4472 | 0.4486 |

This set was selected nonrandomly. Its results are near chance and cannot represent the entire T3 library. A few old-protocol optimization endpoints still had large energy differences; agreement in aggregate metrics does not imply agreement in every molecule's energy. See the [smoke comparison](../results/aimnet2-reproduction-20261007/smoke_metrics_comparison.csv).

Both geometries and all seven variants for KIN66 / PLA15 were recomputed. The table below uses the `released_xyz` geometry and `int_b973c` reference; RMSE is in kcal/mol:

| Model | KIN66 RMSE | PLA15 RMSE |
| --- | --- | --- |
| Old wB97M | 62.6472 | 31.0367 |
| Revised member 0 | 3.4937 | 5.3966 |
| Revised four-member average | 3.2245 | 4.7693 |

All 1,134 individual model predictions matched corresponding reference entries. The largest individual rerun energy difference was below 0.009 kcal/mol. Improved quantum interaction-energy accuracy was reproduced, but this does not establish improved experimental affinity or screening rankings. See the [complete reference metrics and geometry definitions](../results/aimnet2-reproduction-20261007/kin_reference_metrics.csv).

## 3. Testing on Our Frozen T3 Data

We sampled from our frozen normalized T3 final-quota subset. Under the strict L1–L4 definitions, we selected up to 24 targets per layer and randomly selected 10 active ligands per target, using a fixed random seed recorded in the protocol summary. L3 had only 20 available targets. The sample comprised 92 target-layer combinations, 90 distinct UniProt IDs, and 920 requested cases. Failed or chemically excluded cases were not replaced.

We retained the original scientific preparation and scoring modules: exact residue-source structures, one-residue padding, ACE/NME caps, and Amber14 hydrogen addition at pH 7.4; fresh ETKDGv3 / MMFF94s conformations; SMINA exhaustiveness 8, seed 1, cpu 4, and one mode; and FIRE ligand optimization with a fixed receptor, at most 1,000 steps, and fmax 0.002 eV/Å. An adapter replaced filesystem paths. The original docking stage was invoked per ligand so that other valid molecules could proceed; molecule-identity rejection checks were retained.

| Layer | Successful systems / systems with defined correlation | Successful ligands | AIMNet Eint mean ρ | SMINA mean ρ |
| --- | --- | --- | --- | --- |
| L1 | 16 / 15 | 129 | −0.0204 | −0.0106 |
| L2 | 17 / 17 | 139 | −0.1445 | −0.0948 |
| L3 | 14 / 14 | 122 | 0.0667 | 0.2165 |
| L4 | 20 / 19 | 157 | −0.0363 | 0.1162 |

Twenty systems had unsupported chemistry and five failed preparation. Of the 670 ligands in the 67 successfully prepared systems, 123 were rejected by docking or identity checks. The final success count was 547 / 920 = 59.5%; 502 optimizations converged, and 45 returned finite scores without convergence. There were no scoring or system-execution errors. Correlations describe only successfully scored samples and cannot be extrapolated to excluded cases. When restricted to converged molecules, the L1–L4 mean correlations were −0.0197, −0.1369, 0.0462, and 0.0488, respectively; the signal remained weak.

We also strictly paired the results with frozen scores from HypSeek `_rk`, LigUnity protein / pocket, DrugCLIP, LiTENCLIP, and ConGLUDe. Each comparison used the same valid ligands within a target, required at least five ligands, and used 10,000 bootstrap resamples across systems. Of the 24 model-layer comparisons, the mean difference, AIMNet minus the existing model, was positive in 13 and negative in 11. **All 95% intervals included zero**, with no consistent advantage across layers. The six existing models were not rerun; NPZ hashes, model molecule order, and label correspondence were checked.

These tests evaluate affinity ranking among active ligands. They do not provide full-library EF, and we did not rerank our complete active / decoy retrieval library. See the [547 new scores](../results/aimnet2-reproduction-20261007/own_t3_scores.csv), [67-system metrics](../results/aimnet2-reproduction-20261007/own_t3_per_target_metrics.csv), and [paired intervals against six models](../results/aimnet2-reproduction-20261007/own_t3_paired_baselines.csv).

## 4. What the Reproduction Supports and Where It Is Limited

The evidence supports the original work's main metrics and conclusions. It does not establish that every input, conformation, and value is identical, or that AIMNet2 has no signal in any task.

- The latest protocol and FEP both used the original safe neighbor-list implementation; the legacy protocol retained full-range Coulomb neighbors. Independent checks on three systems, using NumPy neighbor matrices and direct TorchScript calls, passed with a maximum interaction-energy difference of 0.00158 kcal/mol.
- Across the 268 old and revised archived cases, energy-component identity errors were below 1e−6 kcal/mol. For every finite result, the complex energy after optimization did not exceed its initial value.
- Earlier PDB charge parsing followed the original rule, defaulting to 0 when the field was missing. Latest T3 / FEP used the prepared net charges. These protocol differences were retained for their respective reproductions and should not be attributed solely to model weights.
- A few CASF optimization endpoints had large energy differences, with a maximum interaction-energy difference of about 289 kcal/mol. Single-point recalculation of both runs' saved structures agreed closely with their respective stored values. This supports an explanation involving optimization trajectories or endpoints, but does not prove that every discrepancy has the same cause.
- Original nonfinite results, preparation failures, chemistry exclusions, and identity rejections were retained in the audit. A first-finite retry was used only for the originally failed molecule; optimization results were not selected based on similarity to the reference.
- Exact parameters or input provenance were incomplete for eight historical exploratory CSVs. The external HiQBind reference preparation and weight provenance were also unconfirmed. These were not counted as strictly reproduced experiments.

The original recorded CUDA runtime was cu130; ours was cu126. The complete original docking / RDKit environment was not fully pinned. Our execution used Python 3.11, AIMNet 0.2, PyTorch 2.14, OpenMM 8.6.1 Reference, and PDBFixer 1.12. The [summary record](../results/aimnet2-reproduction-20261007/summary.json) preserves actual versions, commits, weight identities, and errors. Some earlier case receipts recorded only model aliases; later receipts also recorded resolved paths and hashes. The summary distinguishes these evidence types rather than retrospectively claiming that earlier receipts already contained per-case hashes.

## 5. Result Records and Verification

The accompanying [lightweight results directory](../results/aimnet2-reproduction-20261007) contains per-system reference / rerun comparisons, our new scores, paired intervals, quantum-reference metrics, and protocol summaries. It does not include a fully installed execution environment, model weights, all molecular coordinates, or the full T3 library.

Run from the repository root:

```bash
python3 physics/verify_aimnet_reproduction_summary.py
```

The script uses only the Python standard library. It checks equally weighted reference and rerun means from the included per-system records, and recomputes T3 Spearman correlations with tied ranks, energy decomposition, and convergence counts from the 547 new scores. It does not rerun molecular energy calculations or recompute screening AUROC where the underlying molecule scores are not included.

## Summary for Sharing

> We independently reran the main traceable experiments in the linked source repository. The mean FEP correlation was 0.5002, compared with the reported 0.4992. CASF EF1 after optimization was 0.9722 for both interaction and composite scores, confirming that energy reranking did not improve the retrieval-selected candidates. The main trends in the original T3 and smoke experiments were also similar. We additionally tested the pipeline on our frozen T3 sample, where affinity-ranking signals remained weak with no consistent advantage over the paired baselines. Some fresh docking conformations and optimization endpoints differed, so the agreement is in the main metrics and conclusions rather than every individual value.
