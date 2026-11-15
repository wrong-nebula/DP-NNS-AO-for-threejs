"""Ray-traced AO reference with the analytic moving objects at frozen poses.

Same camera, pixels (57,216 at stride 4) and 512 cosine-weighted rays per pixel as reference.py.
Mover transforms, parameters and mesh triangles come from the viewer itself (benchmark/meshes/,
exported by the harness). Rays test the static boxes and the box and sphere movers analytically,
the capsule, cylinder, rounded box and torus by sphere tracing their exact SDFs, and the teapot and
torus knot triangles with tools/trace_tris.cpp. Pixels whose camera ray hits a mover first are not
scored (they show the mover).

  python3 benchmark/reference_movers.py out.npz --tracer ./trace_tris

Writes per pose: ao (reference with movers), affected (pixels whose reference changes vs static),
scored (pixels usable for comparison).
"""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
RADIUS = 1.12
import sys
sys.path.insert(0, str(ROOT))
from mover_cull_model import surface_points  # same camera and pixels

def sdf(kind, k, q):
    if kind == 0:
        d = np.abs(q) - k; return np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(d.max(-1), 0)
    if kind == 1: return np.linalg.norm(q, axis=-1) - k[0]
    if kind == 2:
        q = q.copy(); q[..., 1] -= np.clip(q[..., 1], -k[1], k[1]); return np.linalg.norm(q, axis=-1) - k[0]
    if kind == 3:
        d = np.abs(np.stack([np.linalg.norm(q[..., [0, 2]], axis=-1), q[..., 1]], -1)) - np.array(k)
        return np.minimum(d.max(-1), 0) + np.linalg.norm(np.maximum(d, 0), axis=-1)
    if kind == 4:
        h = np.array(k[:3]) - k[3]; d = np.abs(q) - h
        return np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(d.max(-1), 0) - k[3]
    if kind == 5:
        return np.linalg.norm(np.stack([np.linalg.norm(q[..., :2], axis=-1) - k[0], q[..., 2]], -1), axis=-1) - k[1]

def ray_hits(kind, k, o, d, tmax):
    """o, d: (N,3) in shape-local space. True where the ray hits the shape at t in [0, tmax]."""
    if kind == 0:
        with np.errstate(divide='ignore', invalid='ignore'):
            a = (-np.array(k) - o) / d; b = (np.array(k) - o) / d
        near = np.nanmax(np.minimum(a, b), -1); far = np.nanmin(np.maximum(a, b), -1)
        return (far >= np.maximum(near, 0)) & (np.maximum(near, 0) <= tmax)
    if kind == 1:
        bq = (o * d).sum(-1); c = (o * o).sum(-1) - k[0] ** 2; disc = bq * bq - c
        t0 = -bq - np.sqrt(np.maximum(disc, 0)); t1 = -bq + np.sqrt(np.maximum(disc, 0))
        return (disc >= 0) & (t1 >= 0) & (np.maximum(t0, 0) <= tmax)
    # sphere tracing on the exact SDF
    t = np.zeros(len(o)); hit = np.zeros(len(o), bool); live = np.ones(len(o), bool)
    tmax = np.broadcast_to(np.asarray(tmax, float), (len(o),))
    for _ in range(256):
        idx = np.flatnonzero(live)
        if not len(idx): break
        s = sdf(kind, k, o[idx] + d[idx] * t[idx, None])
        h = s < 1e-5; hit[idx[h]] = True; live[idx[h]] = False
        t[idx[~h]] += np.maximum(s[~h], 1e-5)
        miss = idx[~h]; live[miss[t[miss] > tmax[miss]]] = False
    return hit

def directions(n, samples):
    a = np.argmax(np.abs(n), 1)
    tangent = np.eye(3)[(a + 1) % 3]; bitangent = np.cross(n, tangent)
    u = (np.arange(samples) + .5) / samples; phi = np.arange(samples) * np.pi * (3 - np.sqrt(5)) + .013
    return (tangent[:, None, :] * (np.sqrt(u) * np.cos(phi))[None, :, None] +
            bitangent[:, None, :] * (np.sqrt(u) * np.sin(phi))[None, :, None] + n[:, None, :] * np.sqrt(1 - u)[None, :, None])

def static_hits(origin, dirs, boxes):
    hits = np.zeros(dirs.shape[:2], bool)
    for box in boxes:
        cand = np.flatnonzero(np.all(np.abs(origin - box[:3]) <= box[3:] + RADIUS, axis=1))
        if not len(cand): continue
        with np.errstate(divide='ignore', invalid='ignore'):
            a = (box[:3] - box[3:] - origin[cand, None, :]) / dirs[cand]; b = (box[:3] + box[3:] - origin[cand, None, :]) / dirs[cand]
        near = np.minimum(a, b).max(-1); far = np.maximum(a, b).min(-1)
        hits[cand] |= np.maximum(near, 0) <= np.minimum(far, RADIUS)
    return hits

TRACER = None  # path to the trace_tris binary (tools/trace_tris.cpp), set by --tracer

def trace_tris(tris, o, d, tmax, tmp):
    """Any-hit test of rays against world-space triangles with the C++ tracer."""
    import subprocess
    n = len(o)
    if not n: return np.zeros(0, bool)
    rays = np.concatenate([o, d, np.broadcast_to(np.asarray(tmax, float), (n,))[:, None]], 1).astype(np.float32)
    tris.astype(np.float32).tofile(tmp / 'tris.f32'); rays.tofile(tmp / 'rays.f32')
    subprocess.run([TRACER, str(tmp / 'tris.f32'), str(tmp / 'rays.f32'), str(tmp / 'hits.u8')], check=True, capture_output=True)
    return np.fromfile(tmp / 'hits.u8', np.uint8).astype(bool)

def load_movers(t):
    """Movers at pose t with the viewer's exact transforms (exported to meshes/poses.json)."""
    poses = json.loads((ROOT / 'meshes' / 'poses.json').read_text())
    key = min(poses, key=lambda k: abs(float(k) - t))
    out = []
    for m in poses[key]:
        M = np.array(m['matrix']).reshape(4, 4).T          # three.js elements are column-major
        R, c = M[:3, :3], M[:3, 3]
        if m['type'] == 6:
            local = np.fromfile(ROOT / 'meshes' / (m['name'].replace(' ', '_') + '.f32'), np.float32).reshape(-1, 3)
            tris = (local @ R.T + c).reshape(-1, 9)
            out.append(dict(name=m['name'], kind=6, tris=tris, c=c, R=R, rb=m['radius']))
        else:
            k = m['param']; kind = m['type']
            k = {0: tuple(k[:3]), 1: (k[0],), 2: (k[0], k[1]), 3: (k[0], k[1]), 4: tuple(k[:4]), 5: (k[0], k[1])}[kind]
            out.append(dict(name=m['name'], kind=kind, k=k, c=c, R=R, rb=m['radius']))
    return out

def pose_reference(t, rays, tmp):
    p, n, _ = surface_points()
    boxes = np.array(json.loads((ROOT / 'archive.json').read_text()))
    eye = np.array([18., -4, 7.3])
    ref = np.load(ROOT / 'reference.npz'); ao_static = ref['ao'][np.isfinite(ref['ao'])]
    scored = np.ones(len(p), bool); cand = np.zeros(len(p), bool)
    movers = load_movers(t)
    v = p - eye; L = np.linalg.norm(v, axis=1); dl = v / L[:, None]
    for m in movers:
        c, R, rb = m['c'], m['R'], m['rb']
        cand |= np.linalg.norm(p - c, axis=1) < rb + RADIUS + .05
        # camera ray reaching p: does it hit this mover first? (those pixels show the mover)
        near = np.flatnonzero(np.linalg.norm(np.cross(dl, c - eye), axis=1) < rb + .01)
        if not len(near): continue
        if m['kind'] == 6:
            hit = trace_tris(m['tris'], np.repeat(eye[None], len(near), 0), dl[near], L[near] - 1e-3, tmp)
        else:
            o = (eye - c) @ R
            hit = ray_hits(m['kind'], m['k'], np.repeat(o[None], len(near), 0), dl[near] @ R, L[near] - 1e-3)
        scored[near[hit]] = False
    idx = np.flatnonzero(cand & scored)
    ao = ao_static.copy()
    origin = p[idx] + n[idx] * .002
    for chunk in np.array_split(np.arange(len(idx)), max(1, len(idx) // 512)):
        if not len(chunk): continue
        ids = idx[chunk]; o = origin[chunk]; dirs = directions(n[ids], rays)
        hits = static_hits(o, dirs, boxes)
        for m in movers:
            c, R = m['c'], m['R']
            near = np.linalg.norm(o - c, axis=1) < RADIUS + m['rb'] + .01
            if not near.any(): continue
            sub = np.flatnonzero(near)
            todo = ~hits[sub].reshape(-1)
            if not todo.any(): continue
            h = np.zeros(len(sub) * rays, bool)
            if m['kind'] == 6:
                ow = np.repeat(o[sub][:, None, :], rays, 1).reshape(-1, 3)
                h[todo] = trace_tris(m['tris'], ow[todo], dirs[sub].reshape(-1, 3)[todo], RADIUS, tmp)
            else:
                ol = np.repeat(((o[sub] - c) @ R)[:, None, :], rays, 1).reshape(-1, 3)
                h[todo] = ray_hits(m['kind'], m['k'], ol[todo], (dirs[sub] @ R).reshape(-1, 3)[todo], RADIUS)
            hits[sub] |= h.reshape(len(sub), rays)
        ao[ids] = 1 - hits.mean(1)
    affected = np.abs(ao - ao_static) > 1e-9
    return ao, affected, scored

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('output', type=Path)
    ap.add_argument('--poses', type=float, nargs='+', default=[0., 3.7, 7.4, 11.1, 14.8, 18.5, 22.2, 25.9])
    ap.add_argument('--rays', type=int, default=512); ap.add_argument('--tracer', required=True)
    a = ap.parse_args(); out = {}
    global TRACER; TRACER = a.tracer
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    for t in a.poses:
        ao, affected, scored = pose_reference(t, a.rays, tmp)
        out[f'ao@{t}'] = ao; out[f'affected@{t}'] = affected; out[f'scored@{t}'] = scored
        print(f'pose {t}: {affected.sum():,} pixels changed by movers, {scored.sum():,} scored')
    np.savez_compressed(a.output, **out)

if __name__ == '__main__':
    main()
