"""The "Traced SDF (temporal)" AO estimator of index.html, as a numpy model the accuracy judge runs.

Two functions, each ported line for line to GLSL (trace -> tracedAO() in the AO pass, show -> the blend in the
scene draw):

  trace(n, frame, noise, sample) -> (P,)  one frame's estimate of ambient visibility in [0, 1] per pixel.
      n       (P, 3) unit surface normals
      frame   int, the frame index (0, 1, 2, ...)
      noise   (P,) per-pixel uniform numbers in [0, 1), new every frame (interleaved gradient noise on the GPU)
      sample  sample(offset, live) -> (P,): the scene's signed distance at the pixel's surface point + offset
              (offset (P, 3)), for the pixels where the bool mask `live` (P,) is True; 1e4 elsewhere. Each live
              pixel costs one 3D texture read; the judge counts them.
  show(formula, traced, frames) -> (P,)  the AO the screen shows, from the shipped four-tap formula's AO, the
      accumulated traced mean, and the number of frames in that mean (1 .. 128).

The viewer accumulates trace() as mean += (est - mean) / frames with frames = min(frame + 1, 128).

Rules (the judge checks them before running): only `import numpy as np`; np functions from a short list;
`for name in range(K)` loops with a constant K <= 64 (a fixed-length GLSL loop) and no other loops; boolean
masks and indexing are fine; no positions, files or scene data. Keep it to what a fragment shader can do.
"""
import numpy as np


def trace(n, frame, noise, sample):
    """2 cosine-weighted directions per frame, each sphere-traced up to 1.12 m with steps of at least 7.2 cm.
    Soft visibility per direction: the closest approach min over steps of clamp((d + e) / (4 e), 0, 1), with
    e = min(0.025, 0.1 * t * (w.n)) (t*(w.n): the distance to the pixel's own tangent plane, so its own surface
    never counts). The ramp straddles the seed's hard threshold d < e (from -e to 3e), so its mean stays close to
    the hard test's while each direction carries a continuous value; a ray stops once d < -e (fully blocked)."""
    nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
    # Tangent frame: t = normalize(cross(n, a)), a = x axis unless n is close to it; b = cross(n, t).
    ax = np.where(np.abs(nx) < 0.9, 1.0, 0.0)
    ay = 1.0 - ax
    tx = -nz * ay
    ty = nz * ax
    tz = nx * ay - ny * ax
    tl = np.sqrt(tx * tx + ty * ty + tz * tz)
    tx, ty, tz = tx / tl, ty / tl, tz / tl
    bx = ny * tz - nz * ty
    by = nz * tx - nx * tz
    bz = nx * ty - ny * tx
    g1 = (frame * 0.6180339887) % 1.0
    g2 = (frame * 0.7548776662) % 1.0
    rot = 6.2831853 * noise
    vis = np.zeros_like(noise)
    for i in range(2):
        u = (i + g1) / 2.0
        phi = i * 2.39996323 + 6.2831853 * g2 + rot
        r = np.sqrt(u)
        z = np.sqrt(1.0 - u)
        wx = tx * (r * np.cos(phi)) + bx * (r * np.sin(phi)) + nx * z
        wy = ty * (r * np.cos(phi)) + by * (r * np.sin(phi)) + ny * z
        wz = tz * (r * np.cos(phi)) + bz * (r * np.sin(phi)) + nz * z
        cosn = wx * nx + wy * ny + wz * nz
        s = np.full_like(noise, 0.02)
        live = noise >= 0.0
        v = np.ones_like(noise)
        for k in range(16):
            d = sample(np.stack([wx * s, wy * s, wz * s], axis=1), live)
            e = np.minimum(0.025, 0.1 * s * cosn)
            v = np.where(live, np.minimum(v, np.clip((d + e) / (4.0 * e), 0.0, 1.0)), v)
            s = s + np.maximum(d, 0.072)
            live = live & (d >= -e) & (s <= 1.12)
        vis = vis + v
    return vis / 2.0


def show(formula, traced, frames):
    """The formula blended with the traced mean, the traced weight frames / (frames + 8): a fresh history shows
    the formula, a long one the traced mean."""
    return formula + (traced - formula) * (frames / (frames + 8.0))
