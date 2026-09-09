#!/bin/bash
# 同家族 swap 的分发：口袋类走 pocket 版，序列类走 seq 版。
set -u
M=$1
case " conplex conglude sprint " in
  *" $M "*) exec ./run_swap_fam_seq.sh "$@" ;;
  *) exec ./run_swap_fam_pocket.sh "$@" ;;
esac
