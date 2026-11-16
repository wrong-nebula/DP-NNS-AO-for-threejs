#!/usr/bin/env bash
# Interleaved A/B of two bake_sdf binaries (champion vs candidate) with quality guards.
#   benchmark/bench_bake.sh ./bake_champion ./bake_candidate [rounds=7]
# Runs both on the Archive (box JSON) and its 16x16-subdivided dense mesh, alternating every run
# so machine load hits both equally. Prints median totals and the load average before/after.
# Guards (exit 1 on failure): candidate fields are bit-identical to the champion's, random voxels match
# brute force (bake_sdf --validate: every triangle and its winding number, so a wrong-but-consistent
# change can't pass), and eval_ao.py MAE stays within 1e-4 of the shipped numbers (0.0421 original,
# 0.0401 wall correction).
set -euo pipefail
CHAMP=$1; CAND=$2; ROUNDS=${3:-7}
cd "$(dirname "$0")/.."
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
A=(--boxes benchmark/archive.json --lo -1 -1 -1 --span 26 18 8 --res 128 --clip 1.5)
total() { grep -o '"total":[0-9.]*' | cut -d: -f2; }
median() { printf '%s\n' "$@" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
echo "load before: $(uptime | sed 's/.*load averages*: //')"
ca=(); na=(); cd_=(); nd=()
for ((r = 0; r < ROUNDS; r++)); do
  if ((r % 2)); then order=(cand champ); else order=(champ cand); fi
  for who in "${order[@]}"; do
    bin=$CHAMP; [[ $who == cand ]] && bin=$CAND
    t1=$("$bin" "${A[@]}" --out "$TMP/$who.f32" | total)
    t2=$("$bin" "${A[@]}" --subdiv 16 --out "$TMP/$who.dense.f32" | total)
    if [[ $who == champ ]]; then ca+=("$t1"); cd_+=("$t2"); else na+=("$t1"); nd+=("$t2"); fi
  done
done
echo "load after:  $(uptime | sed 's/.*load averages*: //')"
ma=$(median "${ca[@]}"); mb=$(median "${na[@]}"); md=$(median "${cd_[@]}"); me=$(median "${nd[@]}")
printf 'archive  champion %8.1f ms   candidate %8.1f ms   (%+.1f%%)\n' "$ma" "$mb" "$(echo "($mb/$ma-1)*100" | bc -l)"
printf 'dense    champion %8.1f ms   candidate %8.1f ms   (%+.1f%%)\n' "$md" "$me" "$(echo "($me/$md-1)*100" | bc -l)"
ok=1
cmp -s "$TMP/champ.f32" "$TMP/cand.f32" && echo "archive field: bit-identical" || { echo "archive field: DIFFERS"; ok=0; }
cmp -s "$TMP/champ.dense.f32" "$TMP/cand.dense.f32" && echo "dense field:   bit-identical" || { echo "dense field:   DIFFERS"; ok=0; }
# Brute-force check, outside the timed runs. Older binaries without --validate fail here too.
"$CAND" "${A[@]}" --validate 50000 --out "$TMP/v.f32" >/dev/null && echo "archive brute force: 50000 voxels exact" || { echo "archive brute force: FAILED"; ok=0; }
"$CAND" "${A[@]}" --subdiv 16 --validate 1000 --out "$TMP/v.f32" >/dev/null && echo "dense brute force:   1000 voxels exact" || { echo "dense brute force:   FAILED"; ok=0; }
python3 -I benchmark/eval_ao.py "$TMP/cand.f32" --max-mae4 0.0422 --max-mae14 0.0402 || { echo "MAE guard FAILED"; ok=0; }
((ok)) && echo "GUARDS: PASS" || { echo "GUARDS: FAIL"; exit 1; }
