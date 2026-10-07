"""Audit the published per-target summaries and 547 fresh T3 scores, without a GPU."""
import collections
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'results/aimnet2-reproduction-20261007'


def rows(name):
    with (ROOT / name).open(newline='') as handle:
        return list(csv.DictReader(handle))


def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def mean(values):
    a = [x for x in map(number, values) if x is not None]
    return statistics.fmean(a) if a else None


def equal(a, b):
    a, b = number(a), number(b)
    assert (a is None and b is None) or (a is not None and b is not None and abs(a - b) < 1e-10), (a, b)


def ranks(a):
    order = sorted(range(len(a)), key=a.__getitem__)
    result = [0.0] * len(a)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        for k in order[i:j]:
            result[k] = (i + 1 + j) / 2
        i = j
    return result


def rho(x, y):
    if len(x) < 3:
        return None
    a, b = ranks(x), ranks(y)
    am, bm = statistics.fmean(a), statistics.fmean(b)
    den = math.sqrt(sum((v - am) ** 2 for v in a) * sum((v - bm) ** 2 for v in b))
    return sum((v - am) * (w - bm) for v, w in zip(a, b)) / den if den else None


def main():
    summary = json.loads((ROOT / 'summary.json').read_text())
    fep = rows('fep_system_comparison.csv')
    assert len(fep) == 14 and sum(int(x['n_matched']) for x in fep) == 403
    for column, field in [('post_rho_ref', 'mean_post_rho_reference_on_matched'), ('post_rho_rerun', 'mean_post_rho_rerun')]:
        equal(mean(x[column] for x in fep), summary['fep'][field])
    latest = rows('latest_t3_per_target_comparison.csv')
    assert len(latest) == 41 and sum(int(x['n_ligands']) for x in latest) == 403
    for row in rows('latest_t3_layer_comparison.csv'):
        group = [x for x in latest if x['layer'] == row['layer']]
        for field in ['reference_rho', 'fixed_pose_rho', 'fresh_docking_rho']:
            equal(mean(x[field] for x in group), row[field])
    composite = rows('composite_t3_per_target_comparison.csv')
    assert len(composite) == 186
    for row in rows('composite_t3_layer_comparison.csv'):
        group = [x for x in composite if (x['variant'], x['layer']) == (row['variant'], row['layer'])]
        assert len(group) == int(row['n_systems'])
        for field in ['reference_interaction_rho', 'reference_composite_rho', 'rerun_interaction_rho', 'rerun_composite_rho']:
            equal(mean(x[field] for x in group), row[field])
    own = rows('own_t3_scores.csv')
    assert len(own) == 547
    assert len({(x['layer'], x['uniprot'], x['mol_id']) for x in own}) == 547
    groups = collections.defaultdict(list)
    for x in own:
        groups[(x['layer'], x['uniprot'])].append(x)
        decomposition = (float(x['complex_ev']) - float(x['pocket_ev']) - float(x['ligand_ev'])) * 23.060547830619
        assert abs(decomposition - float(x['aimnet_interaction_kcal_mol'])) < 1e-6
        assert float(x['complex_ev']) <= float(x['initial_complex_ev']) + 1e-5
    assert len(groups) == 67 and sum(x['converged'] == 'True' for x in own) == 502
    own_targets = {(x['layer'], x['uniprot']): x for x in rows('own_t3_per_target_metrics.csv')}
    for key, group in groups.items():
        labels = [float(x['paff']) for x in group]
        equal(rho([-float(x['aimnet_interaction_kcal_mol']) for x in group], labels), own_targets[key]['eint_rho'])
    for row in rows('own_t3_layer_metrics.csv'):
        equal(mean(x['eint_rho'] for key, x in own_targets.items() if key[0] == row['layer']), row['mean_eint_rho'])
    paired = rows('own_t3_paired_baselines.csv')
    assert len(paired) == 24
    assert all(float(x['delta_ci95_low']) <= 0 <= float(x['delta_ci95_high']) for x in paired)
    assert sum(float(x['mean_delta']) > 0 for x in paired) == 13
    print('PASS: FEP 14/403, latest T3 41/403, composite 186 systems, own T3 67/547, energy identities and 24 paired intervals.')


if __name__ == '__main__':
    main()
