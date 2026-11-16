"""Field error of point-cloud SDFs against the exact triangle SDF (both 128^3 float32, clip 1.5).

  python3 research/dpnns/field_error.py exact.f32 points_100k.f32 points_300k.f32 ...

Per field: mean / p95 / max |error| over voxels whose exact distance is within the AO reach band
(|d| < 1.2, the largest tap is 1.12), and the share of those voxels whose sign is wrong (|d| > 2 cm).
"""
import sys
import numpy as np

exact = np.fromfile(sys.argv[1], np.float32)
band = np.abs(exact) < 1.2
print(f"voxels in reach band: {band.sum()} of {exact.size}")
print(f"{'field':40s} {'mean_mm':>8s} {'p95_mm':>8s} {'p99_mm':>8s} {'max_mm':>8s} {'sign_err':>9s}")
for path in sys.argv[2:]:
    f = np.fromfile(path, np.float32)
    e = np.abs(f - exact)[band] * 1000
    flip = (np.sign(f) != np.sign(exact)) & band & (np.abs(exact) > .02)
    print(f"{path.split('/')[-1]:40s} {e.mean():8.2f} {np.percentile(e, 95):8.2f} {np.percentile(e, 99):8.2f} {e.max():8.1f} {flip.sum() / band.sum():9.2%}")
