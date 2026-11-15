"""Judge-side check for one bake: check.py KIT OUT PARAMS_JSON

Compares the candidate's field with the original baker's reference field for the cell, and scores
the static AO it would produce against the 512-ray reference (archive cell only; the field is the
same scene). Prints one JSON object:
  pass                 the field is valid and every guard below holds
  visible_mismatch     voxels that differ where either value is below 1.33 m (the AO's 1.075 m
                       reach plus a voxel diagonal: trilinear samples there read them). Must be 0.
  mae14, mae4          static AO MAE (wall correction, original 4-tap) vs the ray-traced reference
"""
import gzip
import json
import sys
from pathlib import Path
import numpy as np

kit, out, params = Path(sys.argv[1]), Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text())
cell = params['cell']
reasons = []
f = out / 'field.f32'
if not f.exists() or f.stat().st_size != 128 ** 3 * 4:
    print(json.dumps({'pass': False, 'reasons': ['field.f32 missing or wrong size']})); sys.exit(0)
cand = np.fromfile(f, np.float32)
ref = np.frombuffer(gzip.decompress((kit / 'refs' / f'{cell}.f32.gz').read_bytes()), np.float32)
if not np.isfinite(cand).all():
    reasons.append('non-finite values')
LIMIT = 1.075 + float(np.sqrt((26 / 128) ** 2 + (18 / 128) ** 2 + (8 / 128) ** 2))
visible = (np.minimum(cand, ref) < LIMIT)
mismatch = int((visible & (cand != ref)).sum())
# Outside the visible band, values may only be raised (never pulled below the band).
lowered = int((~visible & (cand < ref) & (cand < LIMIT)).sum())
if mismatch: reasons.append(f'{mismatch} voxels differ inside the AO-visible band')
sys.path.insert(0, str(kit / 'refs'))
import eval_ao  # noqa: E402
tmp = out / 'check_field.f32'; cand.tofile(tmp)
try:
    scores, _, _ = eval_ao.evaluate(tmp)
finally:
    tmp.unlink()
if scores['mae14'] > 0.0402: reasons.append(f"mae14 {scores['mae14']:.5f} > 0.0402")
if scores['mae4'] > 0.0422: reasons.append(f"mae4 {scores['mae4']:.5f} > 0.0422")
print(json.dumps({'pass': not reasons and not lowered, 'reasons': reasons + ([f'{lowered} voxels lowered'] if lowered else []),
                  'visible_mismatch': mismatch, 'mae14': scores['mae14'], 'mae4': scores['mae4']}))
