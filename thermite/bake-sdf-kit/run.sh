#!/bin/sh
# One timed bake: run.sh BUILD PARAMS OUT SCENE_JSON
# PARAMS holds the cell's key=value lines (subdiv=N). The judge times this whole script.
# Threads: the judge discards a block when the harness is CPU-throttled for more than 2% of it, so
# the bake uses a third of the container's CPU quota (cgroup cpu.max), leaving the rest for the judge and agents,
# never every core the machine shows. Both arms get the same count.
set -eu
BUILD=$1; PARAMS=$2; OUT=$3; SCENE=$4
SUBDIV=$(sed -n 's/^subdiv=//p' "$PARAMS")
CPUS=$(nproc)
if [ -r /sys/fs/cgroup/cpu.max ]; then
  read -r QUOTA PERIOD < /sys/fs/cgroup/cpu.max || true
  if [ "${QUOTA:-max}" != max ] && [ "${PERIOD:-0}" -gt 0 ]; then CPUS=$((QUOTA / PERIOD)); fi
fi
# A third of the quota (2 of 6): 4 threads still tripped the 2% throttle check on a shared host.
THREADS=$((CPUS / 3)); [ "$THREADS" -ge 1 ] || THREADS=1
exec "$BUILD/bake_sdf" --boxes "$SCENE" --lo -1 -1 -1 --span 26 18 8 --res 128 --clip 1.5 \
  --subdiv "$SUBDIV" --threads "$THREADS" --out "$OUT/field.f32" --stats "$OUT/stats.json"
