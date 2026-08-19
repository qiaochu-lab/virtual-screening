"""给 DrugCLIP / BindCLIP 加上 --test-task T3。

做法与之前加 DEKOIS 支持时一致：**从已验证的 test_dekois_target 生成新方法，
只做路径替换**，不改任何模型前向逻辑。这样 T3 与 DEKOIS/DUD-E 走的是
同一条推理路径，横评时不会掺进实现差异。

（当初加 DEKOIS 支持时，DrugCLIP 的结果与 LigUnity 论文里报的
DrugCLIP-on-DEKOIS 基线偏差 0.0%，等于独立验证了这种「路径替换」补丁的正确性。）

T3 的数据布局：
    data/T3_6A/<layer>/<uniprot>/<uniprot>_lig.lmdb
                                /<uniprot>_pocket.lmdb
结果落盘：
    <results_path>/T3/<layer>/<uniprot>/saved_preds.npy + saved_labels.npy
"""
import argparse
import os
import re

B = "/data/work/vs-benchmark"

T3_METHODS = '''
    # ===== 本项目新增：T3 时间外推评测 =====
    # 由 test_dekois_target 路径替换生成，模型前向逻辑逐行相同。
    def test_t3_target(self, layer, target, model, **kwargs):
        _root = getattr(self.args, "t3_root", "./data/T3_6A")
        data_path = f"{_root}/{layer}/{target}/{target}_lig.lmdb"
        mol_dataset = self.load_mols_dataset(data_path, "atoms", "coordinates")
        bsz = 64
        mol_reps, labels = [], []
        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz,
                                               collate_fn=mol_dataset.collater)
        for _, sample in enumerate(mol_data):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["mol_src_distance"]
            et = sample["net_input"]["mol_src_edge_type"]
            st = sample["net_input"]["mol_src_tokens"]
            mol_padding_mask = st.eq(model.mol_model.padding_idx)
            mol_x = model.mol_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.mol_model.gbf(dist, et)
            gbf_result = model.mol_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result.permute(0, 3, 1, 2).contiguous().view(-1, n_node, n_node)
            mol_outputs = model.mol_model.encoder(
                mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias)
            mol_emb = model.mol_project(mol_outputs[0][:, 0, :])
            mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
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
            dist = sample["net_input"]["pocket_src_distance"]
            et = sample["net_input"]["pocket_src_edge_type"]
            st = sample["net_input"]["pocket_src_tokens"]
            pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
            pocket_x = model.pocket_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.pocket_model.gbf(dist, et)
            gbf_result = model.pocket_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result.permute(0, 3, 1, 2).contiguous().view(-1, n_node, n_node)
            pocket_outputs = model.pocket_model.encoder(
                pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias)
            pocket_emb = model.pocket_project(pocket_outputs[0][:, 0, :])
            pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
            pocket_reps.append(pocket_emb.detach().cpu().numpy())
        pocket_reps = np.concatenate(pocket_reps, axis=0)

        res = pocket_reps @ mol_reps.T
        res_single = res.max(axis=0)

        _d = f"{self.args.results_path}/T3/{layer}/{target}"
        os.makedirs(_d, exist_ok=True)
        np.save(f"{_d}/saved_preds.npy", res_single)
        np.save(f"{_d}/saved_labels.npy", np.asarray(labels))
        return res_single, labels

    def test_t3(self, model, **kwargs):
        _root = getattr(self.args, "t3_root", "./data/T3_6A")
        layers = [x for x in sorted(os.listdir(_root)) if os.path.isdir(f"{_root}/{x}")]
        for layer in layers:
            targets = sorted(os.listdir(f"{_root}/{layer}"))
            n_ok, n_fail = 0, 0
            for i, target in enumerate(targets):
                try:
                    self.test_t3_target(layer, target, model)
                    n_ok += 1
                except Exception as e:
                    n_fail += 1
                    print(f"[T3][{layer}] {target} 失败: {type(e).__name__}: {e}", flush=True)
                if (i + 1) % 25 == 0:
                    print(f"[T3][{layer}] {i+1}/{len(targets)}  成功 {n_ok} 失败 {n_fail}",
                          flush=True)
            print(f"[T3][{layer}] 完成：成功 {n_ok} / 失败 {n_fail}", flush=True)
        return
'''


def patch_task(path):
    src = open(path).read()
    if "def test_t3_target" in src:
        return "已有 T3 方法，跳过"
    m = re.search(r"\n    def test_dekois\(self", src)
    if not m:
        return "找不到 test_dekois，未改"
    src = src[:m.start()] + "\n" + T3_METHODS + src[m.start():]
    open(path, "w").write(src)
    return "已插入 test_t3 / test_t3_target"


def patch_test_py(path):
    src = open(path).read()
    changed = []
    if 'args.test_task=="T3"' not in src:
        m = re.search(r'(\s+)elif args\.test_task=="DEKOIS":[^\n]*\n(\s+)([^\n]+)\n', src)
        if not m:
            return "找不到 DEKOIS 分支，未改"
        indent_if, indent_body, body = m.group(1), m.group(2), m.group(3)
        call = re.sub(r"test_dekois", "test_t3", body)
        ins = (f'{indent_if}elif args.test_task=="T3":     # 本项目新增\n'
               f'{indent_body}{call}\n')
        src = src[:m.end()] + ins + src[m.end():]
        changed.append("加了 T3 分支")
    src2 = src.replace('choices=["DUDE", "PCBA", "DEKOIS"]',
                       'choices=["DUDE", "PCBA", "DEKOIS", "T3"]')
    if src2 != src:
        src, _ = src2, changed.append("choices 加了 T3")
    if '--t3-root' not in src:
        m = re.search(r'(\s+)parser\.add_argument\("--test-task"[^\n]*\n', src)
        if m:
            ind = m.group(1)
            src = (src[:m.end()] +
                   f'{ind}parser.add_argument("--t3-root", type=str, default="./data/T3_6A",\n'
                   f'{ind}                    help="T3 数据根目录（本项目新增）")\n' +
                   src[m.end():])
            changed.append("加了 --t3-root")
    open(path, "w").write(src)
    return "；".join(changed) if changed else "无需改动"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="+", default=["DrugCLIP", "BindCLIP"])
    args = ap.parse_args()
    for r in args.repos:
        tp = f"{B}/code/{r}/unimol/tasks/drugclip.py"
        xp = f"{B}/code/{r}/unimol/test.py"
        print(f"[{r}]")
        print(f"  tasks/drugclip.py : {patch_task(tp) if os.path.exists(tp) else '文件不存在'}")
        print(f"  test.py           : {patch_test_py(xp) if os.path.exists(xp) else '文件不存在'}")
