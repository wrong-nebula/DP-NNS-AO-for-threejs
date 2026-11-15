"""Static AO quality guard: emulate the viewer's four-tap SDF shader on the benchmark pixels.

Evaluates the 'Original four-tap SDF' (TAPS 4) and 'SDF wall correction' (FAST) formulas from
index.html at the 57,216 visible points of reference.npz (512-ray reference, same camera), using a
baked field (float32, 128^3, x fastest; stored as half floats like the viewer's texture) and
trilinear sampling. Prints MAE against the reference. Exit code 1 if a --max-* guard fails.

  python3 benchmark/eval_ao.py benchmark/archive.f32
  python3 benchmark/eval_ao.py candidate.f32 --max-mae14 0.0402 --max-mae4 0.0422
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from mover_cull_model import surface_points, static_sampler  # noqa: E402  (same camera and sampler)

R = np.array([.16, .48, .80, 1.12])

def evaluate(field_path):
    p, n, inst = surface_points()
    ref = np.load(ROOT / 'reference.npz')
    ao_ref = ref['ao'][np.isfinite(ref['ao'])]
    surf = ref['surface'][np.isfinite(ref['ao'])]
    assert len(ao_ref) == len(p), (len(ao_ref), len(p))
    # Same half-float storage as the viewer's Data3DTexture.
    tmp = np.fromfile(field_path, np.float32).astype(np.float16).astype(np.float32)
    half_path = Path(field_path).with_suffix('.half_eval.f32'); tmp.tofile(half_path)
    try:
        d = static_sampler(half_path)(p[:, None, :] + n[:, None, :] * R[None, :, None])
    finally:
        half_path.unlink()
    # TAPS == 4
    w = 1 - .55 * np.arange(4) / 3
    occ = np.clip((R - d - .045) / R, 0, 1)
    ao4 = 1 - .92 * (occ @ w) / w.sum()
    # FAST (SDF wall correction)
    taps = np.clip(1 - (d + .045) / R, 0, 1)
    wall = np.isin(surf, [1, 2]).astype(float)
    vertical = wall * (1 - np.abs(n[:, 2]))
    far = .40 + .60 * np.maximum(taps[:, 0], taps[:, 1])
    original = .92 * np.array([1, .8166666667, .6333333333, .45]) / 2.9
    corrected = np.stack([np.full_like(far, .46), np.full_like(far, .24), .21 * far, .16 * far], 1)
    weights = original[None] * (1 - vertical[:, None]) + corrected * vertical[:, None]
    ao14 = 1 - (taps * weights).sum(1)
    out = {}
    for name, ao in (('mae4', ao4), ('mae14', ao14)):
        out[name] = float(np.abs(ao - ao_ref).mean())
        out[name.replace('mae', 'wall_mae')] = float(np.abs(ao - ao_ref)[wall > 0].mean())
    return out, ao4, ao14

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('field', type=Path)
    ap.add_argument('--max-mae4', type=float); ap.add_argument('--max-mae14', type=float)
    ap.add_argument('--against', type=Path, help='second field: also report max per-pixel AO difference')
    a = ap.parse_args()
    out, ao4, ao14 = evaluate(a.field)
    if a.against:
        _, b4, b14 = evaluate(a.against)
        out['max_pixel_diff4'] = float(np.abs(ao4 - b4).max()); out['max_pixel_diff14'] = float(np.abs(ao14 - b14).max())
        out['pixels_diff_gt_1_255'] = int(((np.abs(ao14 - b14) > 1 / 255) | (np.abs(ao4 - b4) > 1 / 255)).sum())
    print(json.dumps(out))
    ok = (a.max_mae4 is None or out['mae4'] <= a.max_mae4) and (a.max_mae14 is None or out['mae14'] <= a.max_mae14)
    sys.exit(0 if ok else 1)

if __name__ == '__main__':
    main()
