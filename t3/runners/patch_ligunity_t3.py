"""给 LigUnity 加 --test-task T3。

与 DrugCLIP/BindCLIP 的补丁同样思路：从它已验证的 test_dekois_target
做路径替换生成，模型前向逻辑逐行不动。

LigUnity 与 DrugCLIP 的两点差别：
  1. 口袋塔要传蛋白序列：pocket_forward(protein_sequences=seq, ...)。
     T3 的靶点 ID 本身就是 UniProt 号，序列直接查我们的 sequences.json。
  2. 官方只存 embedding 不存打分，这里补存 saved_preds.npy，
     以便接入统一评测层。

bsz 改成读命令行参数：T3 分子最大 336 个原子（DEKOIS 才 50），
UniMol 注意力 O(n^2)，沿用写死的 64 会 CUDA OOM（DrugCLIP 上已实测）。
另外逐靶点 empty_cache——1,044 个靶点，碎片会累积。
"""
import re

B = "/data/yicheng/xqc/vs-benchmark"
P = f"{B}/code/LigUnity/unimol/tasks/test_task.py"
X = f"{B}/code/LigUnity/unimol/test.py"

T3 = '''
    # ===== 本项目新增：T3 时间外推评测 =====
    # 由 test_dekois_target 路径替换生成，模型前向逻辑逐行相同。
    def test_t3_target(self, layer, target, model, seq, **kwargs):
        _root = getattr(self.args, "t3_root", f"{self.args.data}/../T3_6A")
        data_path = f"{_root}/{layer}/{target}/{target}_lig.lmdb"
        mol_dataset = self.load_mols_dataset(data_path, "atoms", "coordinates")
        bsz = int(getattr(self.args, "batch_size", 8) or 8)
        mol_reps, labels = [], []
        mol_data = torch.utils.data.DataLoader(mol_dataset, num_workers=4, batch_size=bsz,
                                               collate_fn=mol_dataset.collater)
        for _, sample in enumerate(mol_data):
            sample = unicore.utils.move_to_cuda(sample)
            mol_emb = model.mol_forward(**sample["net_input"])
            mol_reps.append(mol_emb.detach().cpu().numpy())
            labels.extend(sample["target"].detach().cpu().numpy())
        mol_reps = np.concatenate(mol_reps, axis=0)
        labels = np.array(labels, dtype=np.int32)

        data_path = f"{_root}/{layer}/{target}/{target}_pocket.lmdb"
        pocket_dataset = self.load_pockets_dataset(data_path)
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=bsz,
                                                  collate_fn=pocket_dataset.collater)
        pocket_reps = []
        for _, sample in enumerate(pocket_data):
            sample = unicore.utils.move_to_cuda(sample)
            pocket_emb = model.pocket_forward(protein_sequences=seq, **sample["net_input"])
            pocket_reps.append(pocket_emb.detach().cpu().numpy())
        pocket_reps = np.concatenate(pocket_reps, axis=0)

        res = pocket_reps @ mol_reps.T
        res_single = res.max(axis=0)

        _d = f"{self.args.results_path}/T3/{layer}/{target}"
        os.makedirs(_d, exist_ok=True)
        np.save(f"{_d}/saved_preds.npy", res_single)      # 官方不存打分，这里补上
        np.save(f"{_d}/saved_labels.npy", labels)
        del mol_reps, pocket_reps, res
        torch.cuda.empty_cache()
        return res_single, labels

    def test_t3(self, model, **kwargs):
        import json as _json
        _root = getattr(self.args, "t3_root", f"{self.args.data}/../T3_6A")
        _seqs = _json.load(open(f"{B_ROOT}/data/t3/sequences.json"))
        layers = [x for x in sorted(os.listdir(_root)) if os.path.isdir(f"{_root}/{x}")]
        for layer in layers:
            targets = sorted(os.listdir(f"{_root}/{layer}"))
            n_ok = n_fail = 0
            for i, target in enumerate(targets):
                s = (_seqs.get(target) or {}).get("seq")
                if not s:
                    n_fail += 1
                    print(f"[T3][{layer}] {target} 跳过：无序列", flush=True)
                    continue
                try:
                    self.test_t3_target(layer, target, model, s)
                    n_ok += 1
                except Exception as e:
                    n_fail += 1
                    torch.cuda.empty_cache()
                    print(f"[T3][{layer}] {target} 失败: {type(e).__name__}: {e}", flush=True)
                if (i + 1) % 25 == 0:
                    print(f"[T3][{layer}] {i+1}/{len(targets)} 成功 {n_ok} 失败 {n_fail}",
                          flush=True)
            print(f"[T3][{layer}] 完成：成功 {n_ok} / 失败 {n_fail}", flush=True)
        return
'''.replace("B_ROOT", repr(B))

s = open(P).read()
if "def test_t3_target" in s:
    print("test_task.py: 已有 T3 方法，跳过")
else:
    m = re.search(r"\n    def test_dekois\(self", s)
    assert m, "找不到 test_dekois"
    s = s[:m.start()] + "\n" + T3 + s[m.start():]
    open(P, "w").write(s)
    print("test_task.py: 已插入 test_t3 / test_t3_target")

x = open(X).read()
if 'args.test_task == "T3"' not in x:
    m = re.search(r'(\s+)elif args\.test_task == "DEKOIS":\n(\s+)task\.test_dekois\(model\)\n', x)
    assert m, "找不到 DEKOIS 分支"
    ins = (f'{m.group(1)}elif args.test_task == "T3":     # 本项目新增\n'
           f'{m.group(2)}task.test_t3(model)\n')
    x = x[:m.end()] + ins + x[m.end():]
if '--t3-root' not in x:
    m = re.search(r'(\s+)parser\.add_argument\("--test-task"[^\n]*\n', x)
    if m:
        ind = m.group(1)
        x = (x[:m.end()] +
             f'{ind}parser.add_argument("--t3-root", type=str,\n'
             f'{ind}                    default="{B}/data/T3_6A",\n'
             f'{ind}                    help="T3 数据根目录（本项目新增）")\n' + x[m.end():])
x = re.sub(r'(choices=\[[^\]]*"DEKOIS"[^\]]*)\]', r'\1, "T3"]', x)
open(X, "w").write(x)
print("test.py: 已加 T3 分支 / --t3-root / choices")
