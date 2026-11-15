"""CPU model of the viewer's moving-object culling on the benchmark pixels.

Counts how many (pixel, mover) and (pixel, mover, tap) evaluations the AO shader
performs with the current tests, and how many tighter but still exact tests would
leave. Pixels are the 57,216 static-surface points from reference.py's camera.
Mover poses follow index.html; bounding radii and local boxes are approximations
of the three.js geometries (see MOVERS). Usage: python3 mover_cull_model.py [field.f32]
"""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
R = np.array([.16, .48, .80, 1.12])
W, H, STRIDE = 1280, 720, 4
EYE = np.array([18., -4, 7.3]); TARGET = np.array([10.8, 8.8, 1.4])

def unit(v): return v / np.linalg.norm(v, axis=-1, keepdims=True)

def surface_points():
    boxes = np.array(json.loads((ROOT / 'archive.json').read_text()))
    x, y = np.meshgrid(np.arange(2, W, STRIDE), np.arange(2, H, STRIDE))
    f = unit(TARGET - EYE); r = unit(np.cross(f, [0, 0, 1])); u = np.cross(r, f)
    s = np.tan(np.deg2rad(37 / 2))
    ray = unit(f + (x.ravel()[:, None] + .5 - W / 2) * 2 / H * s * r + (H / 2 - y.ravel()[:, None] - .5) * 2 / H * s * u)
    t = np.full(len(ray), np.inf); ax = np.zeros(len(ray), int); surf = np.full(len(ray), -1)
    for i, b in enumerate(boxes):
        c, h = b[:3], b[3:]
        with np.errstate(divide='ignore', invalid='ignore'):
            a1 = (c - h - EYE) / ray; a2 = (c + h - EYE) / ray
        lo = np.minimum(a1, a2); near = lo.max(-1); far = np.maximum(a1, a2).min(-1)
        hit = (near >= .1) & (far >= near) & (near < t)
        t[hit], ax[hit], surf[hit] = near[hit], lo.argmax(-1)[hit], i
    v = surf >= 0
    p = EYE + ray[v] * t[v, None]; a = ax[v]
    n = np.eye(3)[a] * (-np.sign(ray[v][np.arange(len(a)), a]))[:, None]
    return p, n, boxes[surf[v]]

def rot(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    X = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]); Y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Z @ Y @ X  # three.js Euler order 'ZYX' as used by index.html

S, C = np.sin, np.cos
# name, bounding radius, local half extents (None = sphere only), convex (self-skip), pose
MOVERS = [
    ('crate', .780, (.45, .45, .45), 1, lambda t: (15 + 6 * S(t * .35), 1.4, .45, 0, 0, t * .6)),
    ('pillar', 1.087, (.3, .3, 1), 1, lambda t: (9.8 + 1.05 * C(t * .5), 8.6 + 1.05 * S(t * .5), 1, 0, 0, -t * .5)),
    ('floating box', .751, (.5, .5, .25), 1, lambda t: (17, 6.5, 1.3 + .7 * S(t * .8), 0, 0, t * .4)),
    ('plank', 1.048, (1, .3, .08), 1, lambda t: (21.8, 6.6, 1, 0, 0, t * .7)),
    ('sphere', .401, (.4, .4, .4), 1, lambda t: (4.5 + 2.5 * S(t * .4), 2, .4, 0, (2.5 * S(t * .4)) / .4, 0)),
    ('rounded box', .455, (.35, .25, .3), 1, lambda t: (6.5, 3.6, .75, t * .7, t * .5, 0)),
    ('capsule', .571, (.22, .57, .22), 0, lambda t: (22.6, 3.3, .9, t * .8, 0, t * .3)),
    ('cylinder', .553, (.32, .45, .32), 0, lambda t: (16.9, 4 + 1.1 * S(t * .45), .45, np.pi / 2, 0, 0)),
    ('torus', .561, (.56, .56, .14), 0, lambda t: (12.45, 4.6, .85, np.pi / 2, 0, t * .6)),
    ('teapot', .48, None, 0, lambda t: (13 + 2.2 * S(t * .3), 11.7, 2.735, 0, 0, t * .5)),
    ('torus knot', .55, None, 0, lambda t: (15, 6, 1.4 + .2 * S(t * .9), t * .4, 0, t * .6)),
]

def sdbox(q, h):
    d = np.abs(q) - h
    return np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(d.max(-1), 0)

def static_sampler(path):
    f = np.fromfile(path, np.float32).reshape(128, 128, 128)  # z, y, x
    lo = np.array([-1., -1, -1]); span = np.array([26., 18, 8])
    def sample(pos):
        g = (pos - lo) / span * 128 - .5
        out = np.any((pos - lo) / span < 0, -1) | np.any((pos - lo) / span > 1, -1)
        g = np.clip(g, 0, 127 - 1e-6); i0 = np.floor(g).astype(int); w = g - i0; i1 = np.minimum(i0 + 1, 127)
        val = 0
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    ix = i1[..., 0] if dx else i0[..., 0]; iy = i1[..., 1] if dy else i0[..., 1]; iz = i1[..., 2] if dz else i0[..., 2]
                    wt = (w[..., 0] if dx else 1 - w[..., 0]) * (w[..., 1] if dy else 1 - w[..., 1]) * (w[..., 2] if dz else 1 - w[..., 2])
                    val = val + wt * f[iz, iy, ix]
        return np.where(out, 1e4, val)
    return sample

def main():
    p, n, inst = surface_points()
    taps = p[:, None, :] + n[:, None, :] * R[None, :, None]       # (P,4,3)
    sample = static_sampler(sys.argv[1] if len(sys.argv) > 1 else ROOT / 'archive.f32')
    ds = sample(taps)                                             # static distance per tap
    useful = R - .045                                             # a mover only matters at tap k if its distance < R_k - .045
    P = len(p); rows = []
    tot = dict(inst=0, sphere=0, sphere_tap=0, obb=0, obb_tap=0, obb_tap_static=0, cell=0)
    times = np.linspace(0, 30, 16)
    for t in times:
        for name, rb, half, convex, pose in MOVERS:
            x, y, z, rx, ry, rz = pose(t); c = np.array([x, y, z]); M = rot(rx, ry, rz)
            # 1. per-instance (vertex shader): mover sphere within 2.25 of the instance's world AABB
            ih = inst[:, 3:]; ic = inst[:, :3]
            m_inst = np.linalg.norm(np.maximum(np.abs(c - ic) - ih, 0), axis=-1) - rb < 2.25
            # world-grid cell variant: 1 m cells, mover sphere inflated by 1.12 overlaps p's cell
            cell_lo = np.floor(p); m_cell = np.linalg.norm(np.maximum(np.abs(c - (cell_lo + .5)) - .5, 0), axis=-1) - rb < 1.12
            # 2. current per-pixel test: farthest tap vs bounding sphere
            m_sph = m_inst & (np.linalg.norm(taps[:, 3] - c, axis=-1) - rb < 1.12 - .044)
            # 3. per-tap sphere test
            dsph = np.linalg.norm(taps - c, axis=-1) - rb               # lower bound on mover distance per tap
            t_sph = m_sph[:, None] & (dsph < useful)
            # 4. oriented box (lower bound for every shape inside it), farthest tap and per tap
            if half is not None:
                q = (taps - c) @ M                                      # world -> local
                dobb = np.maximum(sdbox(q, np.array(half)), dsph)
            else:
                dobb = dsph
            m_obb = m_sph & (dobb[:, 3] < 1.12 - .044)
            t_obb = m_obb[:, None] & (dobb < useful)
            # 5. skip a tap if the static distance there is already <= the mover's lower bound
            t_obb_s = t_obb & (dobb < ds)
            tot['inst'] += m_inst.sum(); tot['cell'] += m_cell.sum(); tot['sphere'] += m_sph.sum()
            tot['sphere_tap'] += t_sph.sum(); tot['obb'] += m_obb.sum(); tot['obb_tap'] += t_obb.sum(); tot['obb_tap_static'] += t_obb_s.sum()
    nT = len(times); ref = tot['sphere'] * 4
    print(f'{P:,} pixels, {len(MOVERS)} movers, {nT} poses')
    print(f"per-pixel candidates after instance mask (loop iterations doing the sphere test): {tot['inst']/nT/P:.2f}")
    print(f"  ... with a 1 m world-cell mask instead:                                     {tot['cell']/nT/P:.2f}")
    print(f"movers evaluated per pixel, current sphere test:  {tot['sphere']/nT/P:.3f}  ({tot['sphere']/nT:,.0f} pixel-movers)")
    print(f"movers evaluated per pixel, + oriented-box test:  {tot['obb']/nT/P:.3f}  ({tot['obb']/tot['sphere']:.0%} of current)")
    print(f"tap evaluations (current = 4 per mover):          {ref/nT:,.0f}")
    print(f"  per-tap sphere test:                            {tot['sphere_tap']/nT:,.0f}  ({tot['sphere_tap']/ref:.0%})")
    print(f"  per-tap oriented-box test:                      {tot['obb_tap']/nT:,.0f}  ({tot['obb_tap']/ref:.0%})")
    print(f"  + skip taps where static distance <= bound:     {tot['obb_tap_static']/nT:,.0f}  ({tot['obb_tap_static']/ref:.0%})")

if __name__ == '__main__':
    main()
