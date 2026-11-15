"""Export reference_movers.npz as flat binaries the browser harness can fetch.

For each pose t: ref_movers_{t}.f32 (320x180 grid, top row first, NaN where nothing visible),
and ref_movers_{t}.u8 with bit 0 = scored, bit 1 = affected by movers. Also ref_static.f32.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
src = np.load(sys.argv[1] if len(sys.argv) > 1 else ROOT / 'reference_movers.npz')
static = np.load(ROOT / 'reference.npz')['ao']
vis = np.isfinite(static)
static.astype(np.float32).tofile(ROOT / 'ref_static.f32')
for key in src.files:
    if not key.startswith('ao@'): continue
    t = key[3:]
    grid = np.full(static.shape, np.nan, np.float32); grid[vis] = src[key]
    flags = np.zeros(static.shape, np.uint8)
    flags[vis] = src[f'scored@{t}'].astype(np.uint8) | (src[f'affected@{t}'].astype(np.uint8) << 1)
    grid.tofile(ROOT / f'ref_movers_{t}.f32'); flags.tofile(ROOT / f'ref_movers_{t}.u8')
    print(t, int(((flags & 3) == 3).sum()), 'scored pixels affected by movers')
