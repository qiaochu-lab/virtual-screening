"""Strict pairing: restrict the retrieval models to the same 10 ligands
AIMNet2 scored, and compare per target.

Why the ligands must be restricted
-------------------------------------
AIMNet2 only scored 10 actives per target, while our T2 uses all actives
within a target (median 24-118). Comparing the two sides' rho directly would
be comparing a correlation over 10 points against one over 100 -- the noise
level differs by an order of magnitude, and the two sides' ligand sets
differ too. A fair comparison needs **the same target, the same ligands**.

Conventions
-------------
- AIMNet2's composite score is better when lower -> negated when correlating
  with pAff
- Molecules are aligned by InChIKey, not by SMILES string or index
- The position of a model's score is decided by T3_model_order.csv, which
  tells whether to read jsonl_pos or lmdb_pos; targets marked FAIL are
  dropped outright (not guessed)
- Per-target paired Wilcoxon test

⚠️ Full set only, and it cannot be moved to the 350-quota subset
-----------------------------------------------------------------
AIMNet2 scored 93 targets; only **30** of them are in the subset, and per
layer that is L1 3, L2 12, L3 7, L4 8. Three targets cannot carry a paired
test, so this comparison is inherently a full-set statement -- the same
situation as the target-class reversal in tasks/T3-time-split.md. Label it
that way wherever it is quoted; do not add --subset here expecting an answer.
"""
import csv, gzip, json, os, collections
import numpy as np
from scipy import stats
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

B = "/data/work/vs-benchmark"
FZ = f"{B}/results/frozen"
# The collaborator's per-ligand scores. This used to read /tmp/aimnet_t3_ligands.csv,
# which stops existing the moment /tmp is cleaned; the committed copy is the one
# every published number came from.
AIMNET_CSV = os.environ.get("AIMNET_CSV", f"{B}/results/T2_aimnet_t3_ligands.csv")
MODELS = ["hypseek_rk", "ligunity_protein_ranking", "ligunity_pocket_ranking",
          "litenclip", "drugclip", "conglude"]


def ikey(smi, cache={}):
    if smi in cache:
        return cache[smi]
    m = Chem.MolFromSmiles(smi)
    try:
        k = Chem.MolToInchiKey(m) if m is not None else ""
    except Exception:
        k = ""
    cache[smi] = k
    return k


# ---- AIMNet2 side ----
ai = collections.defaultdict(list)          # (layer, up) -> [(ikey, paff, -composite, -smina)]
for r in csv.DictReader(open(AIMNET_CSV)):
    k = ikey(r["smiles"])
    if not k:
        continue
    try:
        comp = -float(r["composite"])       # lower is better -> negate
        pa = float(r["paff"])
    except (ValueError, TypeError):
        continue
    sm = None
    try:
        sm = -float(r["smina"])
    except (ValueError, TypeError):
        pass
    ai[(r["layer"], r["uniprot"])].append((k, pa, comp, sm))
print(f"AIMNet2: {len(ai)} 个靶点")

# ---- molecule table & order table ----
mol = {}                                    # mol_id -> ikey
with gzip.open(f"{FZ}/T3_molecules.csv.gz", "rt") as f:
    for r in csv.DictReader(f):
        mol[r["mol_id"]] = r["inchikey"]

order_used = collections.defaultdict(dict)  # model -> (layer,up) -> jsonl|lmdb|FAIL
for r in csv.DictReader(open(f"{FZ}/T3_model_order.csv")):
    order_used[r["model"]][(r["layer"], r["uniprot"])] = r["order_used"]

need = set(ai)
idx = collections.defaultdict(dict)         # (layer,up) -> ikey -> (jsonl_pos, lmdb_pos)
for L in ("L1", "L2", "L3", "L4"):
    with gzip.open(f"{FZ}/T3_index_{L}.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            k = (r["layer"], r["uniprot"])
            if k not in need or r["label"] != "1":
                continue
            ik = mol.get(r["mol_id"], "")
            if ik:
                idx[k][ik] = (int(r["jsonl_pos"]), int(r["lmdb_pos"]))
print(f"索引表覆盖 {len(idx)} 个靶点")

# ---- per model, per target ----
res = collections.defaultdict(lambda: collections.defaultdict(list))
n_skip = collections.Counter()
for m in MODELS:
    for (L, up), lig in sorted(ai.items()):
        ou = order_used[m].get((L, up))
        if ou not in ("jsonl", "lmdb"):
            n_skip[(m, "order")] += 1
            continue
        d = f"{B}/results/t3_raw/{m}/T3/{L}"
        if not os.path.isdir(d):
            d = f"{B}/results/t3/{m}/{L}"
        try:
            p = np.load(f"{d}/{up}/saved_preds.npy").reshape(-1)
        except Exception:
            n_skip[(m, "noscore")] += 1
            continue
        col = 0 if ou == "jsonl" else 1
        pairs = []
        for ik, pa, comp, sm in lig:
            pos = idx[(L, up)].get(ik)
            if pos is None or pos[col] < 0 or pos[col] >= len(p):
                continue
            pairs.append((float(p[pos[col]]), pa, comp))
        if len(pairs) < 5:
            n_skip[(m, "few")] += 1
            continue
        sc = np.array([x[0] for x in pairs]); pa = np.array([x[1] for x in pairs])
        co = np.array([x[2] for x in pairs])
        if np.std(sc) == 0 or np.std(pa) == 0 or np.std(co) == 0:
            continue
        r_model = stats.spearmanr(sc, pa).statistic
        r_ai = stats.spearmanr(co, pa).statistic
        if np.isfinite(r_model) and np.isfinite(r_ai):
            res[m][L].append((r_model, r_ai, len(pairs)))

print("\n同靶点同配体的逐靶点配对：检索模型 vs AIMNet2 复合分")
print("=" * 96)
print("%-26s %-4s %5s %10s %10s %10s %10s %8s" %
      ("模型", "层", "靶点", "检索 ρ", "AIMNet2 ρ", "Δ", "配对 p", "检索胜"))
print("-" * 96)
rows = [["model", "layer", "n_targets", "median_n_ligands", "retrieval_rho",
         "aimnet_rho", "delta", "wilcoxon_p", "retrieval_wins"]]
for m in MODELS:
    for L in ("L1", "L2", "L3", "L4"):
        v = res[m][L]
        if len(v) < 5:
            continue
        a = np.array([x[0] for x in v]); b = np.array([x[1] for x in v])
        nl = int(np.median([x[2] for x in v]))
        try:
            w = stats.wilcoxon(a, b).pvalue
        except Exception:
            w = float("nan")
        print("%-26s %-4s %5d %10.3f %10.3f %+10.3f %10.4g %7d/%d" %
              (m, L, len(v), a.mean(), b.mean(), a.mean()-b.mean(), w,
               int((a > b).sum()), len(v)))
        rows.append([m, L, len(v), nl, f"{a.mean():.4f}", f"{b.mean():.4f}",
                     f"{a.mean()-b.mean():.4f}", f"{w:.4g}", int((a > b).sum())])
    print()
out = f"{B}/results/export/T2_paired_vs_aimnet.csv"
with open(out, "w", newline="") as f:
    csv.writer(f).writerows(rows)
print("-" * 96)
if n_skip:
    print("跳过：", dict(n_skip))
print(f"写入 {out}")
