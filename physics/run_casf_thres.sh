#!/bin/bash
# CASF 泄漏的干预实验：同一批 285 个复合物，换三档 --protein-similarity-thres 权重。
# 观察性拆分（干净/污染靶点）受靶点难度、展布、标签质量三个混杂影响；
# 这里靶点集合完全不变，只有模型变，所以是干预。
set -u
B=/data/work/vs-benchmark
C=$B/ckpt/ligunity/LigUnity_VS
L=$B/results/logs
export PATH=/data/work/envs/ligunity/bin:$PATH
cd $B/code/LigUnity

G=${1:-4}
for pair in "pocket_ranking_vs_0.8 pocket_ranking pocket_ranking_0.8" \
            "pocket_ranking_vs_0.3 pocket_ranking pocket_ranking_0.3" \
            "protein_ranking_vs_0.8 protein_ranking protein_ranking_0.8" \
            "protein_ranking_vs_0.3 protein_ranking protein_ranking_0.3"; do
  set -- $pair
  ck=$1; arch=$2; out=$3
  echo "[$(date +%H:%M:%S)] $ck -> results/$out (GPU $G)"
  CUDA_VISIBLE_DEVICES=$G bash test.sh CASF "$arch" \
      "$C/$ck/checkpoint_avg_41-50.pt" "$B/results/$out" \
      > "$L/${out}_CASF.log" 2>&1
  n=$(python -c "import json;print(len(json.load(open('$B/results/$out/PDBBind/test_pdbbind_ids.json'))))" 2>/dev/null || echo FAIL)
  echo "    复合物数：$n"
done
echo "[$(date +%H:%M:%S)] 完成"
