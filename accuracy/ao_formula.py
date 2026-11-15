"""The static SDF AO formula, as one function the accuracy campaign may rewrite.

    ao(n, wall, sample) -> visibility in [0, 1] per pixel (1 = unoccluded)

n       (N, 3) unit surface normals, world space (z up)
wall    (N,)   1.0 on the two structural walls, else 0.0
sample  sample(offset) -> (N,) signed distance to the static scene at each pixel's surface point
        plus `offset` ((N, 3) metres, usually built from n). At most 4 calls: each is one 3D
        texture read in the shader. Outside the baked volume it returns 1e4.

Rules (enforced by the judge before the formula runs): plain arithmetic, comparisons, `if`
expressions and these np functions only: clip minimum maximum where abs sqrt exp log power square
tanh sign floor fmin fmax. No imports, loops, comprehensions, attributes other than np.<name>,
names starting with "_", pixel-mixing indexing (only n[:, k]), or more than 96 numeric constants (8 KB of source).
The result must stay portable to a few lines of GLSL.

This seed is the viewer's "SDF wall correction" (index.html, FAST path): taps along the normal at
0.16 / 0.48 / 0.80 / 1.12 m, linear occlusion past a 4.5 cm bias, and weights that shift toward
near taps on structural walls.
"""
import numpy as np


def ao(n, wall, sample):
    r0, r1, r2, r3 = 0.16, 0.48, 0.80, 1.12
    up = (0.0, 0.0, 1.0)
    d0 = sample(n * r0)
    d1 = sample(n * r1)
    d2 = sample(n * r2)
    # Fourth tap: 1.12 m along the normal on floors, tops and undersides; on vertical
    # faces 0.8 m tilted 45 degrees up (0.5657 out, 0.5657 up), to see overhangs.
    # n * n is (0, 0, 1) on horizontal faces, so up * n * n and up * (1 - n * n) pick
    # the per-face-class length without any per-pixel scalar broadcast.
    d3 = sample(n * (0.5657 + 0.5543 * (up * n * n)) + 0.5657 * (up * (1 - n * n)))
    vertical = wall * (1 - np.abs(n[:, 2]))
    k = 0.92 / 2.9
    # Structural walls: refit with GPU-robust terms only (square, linear clamps; no sqrt).
    # The biases put each clamp's kink away from where an open wall's taps sit
    # (d ~ r), so 8-bit trilinear / half-float rounding is not amplified. The
    # constant 0.1383 lifts open walls to 1 (output clip).
    v0 = np.clip(1 - (d0 + 0.1526) / r0, 0, 1)
    v1 = np.clip(1 - (d1 - 0.0907) / r1, 0, 1)
    v2 = np.clip(1 - (d2 - 0.5168) / r2, 0, 1)
    v3 = np.clip(1 - (d3 + 0.1875) / 0.8, 0, 1)
    walls = np.clip(1.1383 - (0.0598 * np.square(v0) + 0.5477 * v1 + 0.038 * v2 + 0.0891 * v3), 0, 1)
    # Floors and tops (n.z > 0.7): champion per-tap bias + power on the 4 normal taps.
    s0 = np.clip(1 - (d0 + 0.024) / r0, 0, 1)
    s1 = np.clip(1 - (d1 + 0.078) / r1, 0, 1)
    s2 = np.clip(1 - d2 / r2, 0, 1)
    s3 = np.clip(1 - (d3 + 0.02) / r3, 0, 1)
    s3 = np.square(np.square(np.square(s3)))
    flat = 1 - k * (np.square(s0) + 0.8166666667 * np.sqrt(s1) + 0.6333333333 * np.sqrt(s2) + 0.45 * s3)
    # Other faces: own bias/weights, linear response on the tilted tap.
    u0 = np.clip(1 - (d0 - 0.0061) / r0, 0, 1)
    u1 = np.clip(1 - (d1 + 0.1163) / r1, 0, 1)
    u2 = np.clip(1 - (d2 + 0.0024) / r2, 0, 1)
    u3 = np.clip(1 - (d3 + 0.2143) / 0.8, 0, 1)
    side = 1 - (0.1337 * np.square(u0) + 0.1146 * np.sqrt(u1) + 0.3459 * np.sqrt(u2) + 0.202 * u3)
    tuned = np.where(n[:, 2] > 0.7, flat, side)
    return walls * vertical + tuned * (1 - vertical)
