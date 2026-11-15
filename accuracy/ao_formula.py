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
names starting with "_", pixel-mixing indexing (only n[:, k]), or more than 48 numeric constants.
The result must stay portable to a few lines of GLSL.

This seed is the viewer's "SDF wall correction" (index.html, FAST path): taps along the normal at
0.16 / 0.48 / 0.80 / 1.12 m, linear occlusion past a 4.5 cm bias, and weights that shift toward
near taps on structural walls.
"""
import numpy as np


def ao(n, wall, sample):
    r0, r1, r2, r3 = 0.16, 0.48, 0.80, 1.12
    t0 = np.clip(1 - (sample(n * r0) + 0.045) / r0, 0, 1)
    t1 = np.clip(1 - (sample(n * r1) + 0.045) / r1, 0, 1)
    t2 = np.clip(1 - (sample(n * r2) + 0.045) / r2, 0, 1)
    t3 = np.clip(1 - (sample(n * r3) + 0.045) / r3, 0, 1)
    vertical = wall * (1 - np.abs(n[:, 2]))
    far = 0.40 + 0.60 * np.maximum(t0, t1)
    k = 0.92 / 2.9
    w0 = k + (0.46 - k) * vertical
    w1 = k * 0.8166666667 + (0.24 - k * 0.8166666667) * vertical
    w2 = k * 0.6333333333 + (0.21 * far - k * 0.6333333333) * vertical
    w3 = k * 0.45 + (0.16 * far - k * 0.45) * vertical
    return 1 - (t0 * w0 + t1 * w1 + t2 * w2 + t3 * w3)
