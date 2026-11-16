#!/usr/bin/env bash
# Timing matrix for pointnn_bench + the exact triangle baker, interleaved round-robin so machine load hits
# every configuration alike. Prints one JSON line per run (with the load average) to stdout.
#   research/dpnns/run_bench.sh ./pointnn_bench ./bake_sdf [rounds=3] > runs.jsonl
set -euo pipefail
PB=$1; BK=$2; ROUNDS=${3:-3}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BOX="$ROOT/benchmark/archive.json"
A=(--boxes "$BOX" --lo -1 -1 -1 --span 26 18 8 --res 128 --clip 1.5)
load() { sysctl -n vm.loadavg | awk '{print $2}'; }
for ((r = 0; r < ROUNDS; r++)); do
  for n in 100000 300000 1000000; do
    for t in 1 8; do
      for cfg in "kdtree 0" "bvh 0" "dpnns 0" "dpnns 32" "dpnns 64"; do
        set -- $cfg
        l=$(load)
        out=$("$PB" "${A[@]}" --points $n --method $1 --grid $2 --threads $t)
        echo "{\"round\":$r,\"target\":$n,\"load\":$l,\"run\":$out}"
      done
    done
  done
  for t in 1 8; do
    for blk in 0 8; do
      l=$(load)
      out=$("$BK" "${A[@]}" --threads $t --block $blk --out /dev/null | tail -1)
      echo "{\"round\":$r,\"target\":0,\"load\":$l,\"baker\":{\"threads\":$t,\"block\":$blk},\"run\":$out}"
    done
  done
done
