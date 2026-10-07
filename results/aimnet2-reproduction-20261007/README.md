# AIMNet2 Reproduction Result Supplement

Read the [standalone report](../../tasks/AIMNet2-reproduction.md)
for experiment scope, methods, reference-versus-rerun comparisons and caveats.
This supplement documents independently executed calculations; it is separate
from the canonical screening model tables.

| Files | What they substantiate |
| --- | --- |
| `summary.json` | Pinned upstream commits, numerical errors, coverage, runtime, model identity and audit outcomes |
| `fep_system_comparison.csv` | All 14 paired systems / 403 ligands, before/after correlations and energy differences |
| `casf_metrics_comparison.csv` | Conditional shortlist EF1 reference and independent rerun results |
| `latest_t3_*comparison.csv` | Original latest T3 versus fixed-pose and fresh-docking reruns; 41 systems / 403 ligands |
| `composite_t3_*comparison.csv` | Old and revised composite protocols; 93 systems / 917 ligands each |
| `smoke_metrics_comparison.csv` | Reference and rerun AUROC on the same nonrandom 250-ligand smoke set |
| `kin_reference_metrics.csv` | Every reference, geometry and model variant in the 81-system quantum-energy test |
| `own_t3_scores.csv` | 547 newly computed successful scores, pAffinity labels, component energies and convergence flags |
| `own_t3_per_target_metrics.csv`, `own_t3_layer_metrics.csv` | Per-system correlations and equal-system layer means on our frozen T3 sample |
| `own_t3_paired_targets.csv`, `own_t3_paired_baselines.csv` | Paired affinity comparisons to six existing models and target-bootstrap intervals |

Correlations compare negative energy to pAffinity within each target/layer
system and average valid system correlations with equal weight. Constant-label
systems remain in coverage but have undefined correlation. CASF EF1 uses the
top one of a 20-ligand retrieval-selected shortlist; it is not full-library EF1.

Run from the repository root:

```bash
python3 physics/verify_aimnet_reproduction_summary.py
```

The verifier reconstructs the published per-system means and our new T3
correlations from the included records. It checks unique molecule keys,
energy decomposition, optimizer energy changes and convergence counts. It
does not recompute molecular energies or screening AUROC without the full
underlying molecule scores.

These files do not contain installed environments, model weights, the full
T3 library, or all atomic coordinate inputs. Reference result tables were used
after scoring for comparison, not substituted for independently computed
energies. The six existing T3 models and the Boltz predictions were not rerun;
CASF LigUnity embeddings were regenerated separately. Originally nonfinite
scores retain the documented first-finite retry rule. Successful finite
nonconverged scores were not replaced to make them agree with a reference.
