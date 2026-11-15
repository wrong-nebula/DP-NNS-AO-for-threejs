#!/bin/sh
# One timed bake: run.sh BUILD PARAMS OUT SCENE_JSON
# PARAMS holds the cell's key=value lines (subdiv=N). The judge times this whole script.
set -eu
BUILD=$1; PARAMS=$2; OUT=$3; SCENE=$4
SUBDIV=$(sed -n 's/^subdiv=//p' "$PARAMS")
exec "$BUILD/bake_sdf" --boxes "$SCENE" --lo -1 -1 -1 --span 26 18 8 --res 128 --clip 1.5 \
  --subdiv "$SUBDIV" --out "$OUT/field.f32" --stats "$OUT/stats.json"
