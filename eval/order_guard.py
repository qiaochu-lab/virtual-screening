"""Molecule order guard.

Any script that joins a model's per-molecule score array with information
outside that score (affinity, assay type, contamination flags) must pass
through this check first.

Background: `build_t3_unimol.py` uses `str(i)` as the LMDB key, and reading
it back by cursor on the model side yields lexicographic order
0,1,10,100,...; earlier analysis scripts read by numeric index instead, so a
model's k-th score row got matched to the wrong molecule. This bug corrupted
T2's conclusions twice, silently, with no error raised.

Usage (one line at the top of a script):
    from order_guard import assert_cursor_order
    assert_cursor_order()

Command line:
    python order_guard.py        # exits 0 on pass, 1 on failure
"""
import json, os, pickle, sys
import lmdb, numpy as np

B = "/data/work/vs"


class MoleculeOrderError(RuntimeError):
    pass


def check(layers=("L1", "L4"), model="ligunity_protein_ranking", n_targets=40):
    """Return {layer: agreement rate}. Agreement rate = of the positions
    saved_labels marks as active, how many are actually active on the SMILES
    read back in cursor order."""
    out = {}
    for L in layers:
        p = f"{B}/data/t3/eval/{L}.jsonl"
        root = f"{B}/results/t3_raw/{model}/T3/{L}"
        if not (os.path.exists(p) and os.path.isdir(root)):
            continue
        ev = {json.loads(x)["uniprot"]: json.loads(x) for x in open(p)}
        tot = hit = n = 0
        for up in sorted(os.listdir(root))[:n_targets]:
            lp = f"{root}/{up}/saved_labels.npy"
            dp = f"{B}/data/T3_6A/{L}/{up}/{up}_lig.lmdb"
            rec = ev.get(up)
            if rec is None or not (os.path.exists(lp) and os.path.exists(dp)):
                continue
            y = np.load(lp)
            e = lmdb.open(dp, subdir=False, readonly=True, lock=False)
            smis = []
            with e.begin() as t:
                for _k, v in t.cursor():          # cursor order, not numeric order
                    smis.append(pickle.loads(v)["smi"])
            e.close()
            if len(smis) != len(y):
                continue
            aset = {m["smiles"] for m in rec["actives"]}
            ypos = set(np.where(y == 1)[0].tolist())
            tot += len(ypos)
            hit += len({i for i, s in enumerate(smis) if s in aset} & ypos)
            n += 1
        if tot:
            out[L] = (hit / tot, n, hit, tot)
    return out


def assert_cursor_order(min_agree=0.95, **kw):
    r = check(**kw)
    if not r:
        raise MoleculeOrderError(
            "顺序守卫无法运行：找不到 eval jsonl 或参考模型的 T3 结果。"
            "不要在没验证的情况下拼接逐分子数据。")
    bad = {L: v for L, v in r.items() if v[0] < min_agree}
    if bad:
        raise MoleculeOrderError(
            "分子顺序对不上，拒绝继续："
            + "; ".join(f"{L} 吻合 {v[0]:.1%} ({v[2]}/{v[3]}, {v[1]} 个靶点)"
                        for L, v in bad.items())
            + "。按数值下标读 LMDB 会得到约 10%，按游标序应得约 99.8%。")
    return r


if __name__ == "__main__":
    try:
        r = assert_cursor_order()
    except MoleculeOrderError as e:
        print("✗", e); sys.exit(1)
    for L, (a, n, hit, tot) in r.items():
        print(f"{L}: {n} 个靶点，active 位置吻合 {hit}/{tot} = {a:.1%}")
    print("✓ 顺序守卫通过")
