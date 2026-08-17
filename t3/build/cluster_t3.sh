#!/bin/bash
set -e
B=/data/yicheng/xqc/vs-benchmark
M=/data/yicheng/anconda/envs/dplm_surf_tools/bin/mmseqs
W=$B/data/t3/cluster
mkdir -p $W/tmp
/data/yicheng/xqc/envs/ligunity/bin/python -c "
import json
s = json.load(open('$B/data/t3/sequences.json'))
with open('$W/t3.fasta','w') as f:
    for u,d in s.items():
        f.write('>%s\n%s\n' % (u, d['seq']))
print('写出 %d 条序列' % len(s))
"
# 40% 序列一致性、80% 覆盖度聚类 —— 与 T3 分层用的 CD-HIT 40% 口径一致
$M easy-cluster $W/t3.fasta $W/t3_40 $W/tmp \
  --min-seq-id 0.4 -c 0.8 --cov-mode 1 -v 1
wc -l $W/t3_40_cluster.tsv
