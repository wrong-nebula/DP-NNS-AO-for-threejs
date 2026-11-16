# Hemisphere-traced SDF AO (experiment)

[traced.html](traced.html) replaces the four fixed taps with true hemisphere sampling of the baked field. Per pixel:
- **Directions:** cosine-weighted directions are each sphere-traced through the 128³ SDF up to the AO radius (1.12 m).
- **Blocking:** a direction is blocked when the field drops below `min(0.025, 0.1 · t · (w·n))`. Here `t · (w·n)` is the distance to the pixel's own tangent plane, which keeps the pixel's own surface from counting as a hit.
- **AO:** the unblocked fraction. This is the quantity the 512-ray reference measures, estimated from the field instead of from exact geometry.

Directions are stratified and rotated per pixel, and offset from frame to frame, so a still camera accumulates an ever larger stratified set.

## Accuracy

The three parameters (hit threshold, tangent-plane factor, minimum step) were fitted with a CPU model that filters the field as the GPU does. The model used the six camera views of the accuracy campaign: three for fitting and three held out.

| CPU model, error vs ray-traced | Fit views | Held-out views |
|---|---:|---:|
| Shipped four-tap formula | 0.0290 | 0.0265 |
| Traced, 16 directions | 0.0321 | 0.0292 |
| Traced, 64 directions | 0.0216 | 0.0191 |
| Traced, 256 directions (converged) | 0.0191 | **0.0167** (−37%) |

The GPU (main view, 1280 × 720, Three.js r187 WebGL2, Apple M1) confirms it:

| Directions per frame | AO cost per frame | Error, one frame | Error, accumulated to 64 directions |
|---|---:|---:|---:|
| Four-tap formula (current) | 0.11 ms | 0.0270 | — (deterministic) |
| 2 (×32 frames) | 0.94 ms | 0.0998 | 0.0231 |
| 4 (×16 frames) | 1.79 ms | 0.0613 | 0.0225 |
| 8 (×8 frames) | 3.77 ms | 0.0424 | 0.0221 |
| 16 (×4 frames) | 7.40 ms | 0.0300 | 0.0216 |
| 64 (one frame) | 29.3 ms | 0.0198 | — |

The AO cost is the frame's wall-clock time over AO off: p10 of 30 interleaved rounds, each a batch of 20 frames closed by a readback. One earlier variant took each frame's directions as a stride through a single Fibonacci set. Its accumulated error was 0.0199, but its single frames were badly stratified (0.09 at 8 directions), so it was dropped.

## What it means

- **Accuracy:** converged, traced AO has 15–37% less error than the fitted four-tap formula, and it holds on the held-out views. It also needs no per-scene fitting beyond three generic parameters.
- **Cost:** each traced direction costs about 0.47 ms per frame at 720p on an M1, about 4× the whole four-tap formula. A single frame only matches the formula from 16 directions upward (7.4 ms), so the gain depends on accumulating over time. With 2 directions per frame (about 1 ms), a still camera reaches 0.023 after about half a second at 60 fps.
- **What the viewer would need:** temporal accumulation with reprojection. Static AO depends only on the world position, so reprojected history stays valid for the static scene. Three things are still needed:
  - a disocclusion test on depth and normal;
  - a history cap;
  - a faster history, or the current per-frame mover term, near moving objects.

  ThreeRT's real-time diffuse pass (`dtemporal` in its `live.wgsl`) is a worked example of this pattern.

## In the viewer: "Traced SDF (temporal)"

`index.html` now has this as a mode. The setup:
- **Directions:** 2 per pixel per frame. The per-pixel rotation also changes every frame, which gives more independent samples than a fixed rotation: 0.0209 against 0.0213 at 64 frames.
- **Marching:** 16 steps of at least 6 cm, as accurate as 24 steps of 4 cm on the held-out views (0.0177 against 0.0176).
- **History:**
  - It is reprojected from the previous camera and read with depth- and normal-checked bilinear taps.
  - The AO is stored in a float target: half floats rounded away the small updates of a long history, and the error climbed from 0.0201 to 0.0257 between 64 and 256 frames.
  - The fetch snaps to texel centres when the camera is still; otherwise the bilinear fetch re-blurred the history every frame.
- **Cap:** 128 frames on a still camera, 32 while it moves.
- **Moving objects:** their occlusion is the per-frame formula term, and their own surfaces use the formula.

| Main view, Apple M1, 1280 × 720 | AO error |
|---|---:|
| Formula (current wall correction) | 0.0270 |
| Traced, still: 1 / 16 / 32 / 64 / 256 frames | 0.0993 / 0.0287 / 0.0218 / 0.0191 / **0.0179** |
| Traced, arriving after 60 frames of a 0.25° / 1° per frame orbit | 0.0180 / 0.0220 |
| Traced, arriving after a 3° / 10° per frame orbit | 0.0270 / 0.0415 |
| With moving objects (4 held-out poses, objects animated for 2 s): all pixels | 0.0300 → **0.0225** |
| Near moving objects | 0.0411 → **0.0392** |

**Frame time** (wall clock, p10, readback-closed batches): the formula took 0.31–0.33 ms and traced 1.35–1.43 ms. Profiling an earlier version split the cost as follows:
- the trace, about 0.7 ms;
- the history fetch, about 0.45 ms, halved by skipping zero-weight taps and storing frames in the normal target's alpha;
- writing the extra targets plus the copy to the screen, about 0.45 ms.

### Formula blend for young histories

Without help, a pixel whose history just restarted (a camera cut, a surface revealed behind a moving object) shows a one- or two-direction trace: 0.099 error at one frame, and it takes about 16 frames to match the formula. The shown value is therefore a blend of the wall-correction formula and the traced mean, with traced weight `frames / (frames + k)`. Only the traced mean is stored, so the blend never feeds back.

A linear ramp to pure traced over F frames was tried first. It handed over too early: at F = 8, error jumped from 0.0229 to 0.0362 at the hand-over. The open-ended weight has no jump, and it beats either estimate alone at every history length:

| Main view, AO error | Still: 1 / 16 / 64 / 256 frames | Orbit at 1° / 3° / 10° per frame |
|---|---:|---:|
| Formula | 0.0270 | 0.0270 |
| Traced only | 0.0993 / 0.0287 / 0.0191 / 0.0179 | 0.0220 / 0.0270 / 0.0415 |
| Blend k = 4 | 0.029 / 0.0238 / 0.0181 / 0.0174 | 0.0170 / 0.0191 / 0.0237 |
| **Blend k = 8 (shipped)** | **0.0262 / 0.0218 / 0.0175 / 0.0170** | **0.0167 / 0.0188 / 0.0225** |
| Blend k = 16 | 0.0259 / 0.0207 / 0.0170 / 0.0166 | 0.0173 / 0.0197 / 0.0233 |

- **With moving objects** (4 held-out poses, animated for 2 s): 0.0216 over all pixels and 0.0387 near objects. Without the blend: 0.0225 and 0.0392.
- **Cost:** 4 more texture reads per static pixel, measured at +0.14 ms per frame in an interleaved A/B: 1.44 → 1.58 ms at p10.

### Cutting the cost: settled skip and quarter-rate tiles

A true half-resolution trace would need a separate G-buffer pass, which the forward-rendered viewer doesn't have. Two forward-friendly options were tried instead:
- **Settled skip:** pixels whose history holds `settleFrames` traced frames stop tracing.
- **Quarter-rate tiles:** 8×8-pixel tiles in a 2×2 pattern take turns, so a quarter of the screen traces each frame. Whole tiles keep GPU thread groups uniform, so the skipped work is actually saved.

A just-revealed pixel whose tile isn't tracing shows the formula alone until its turn.

| Main view, AO error | Still: 16 / 64 / 256 frames | Orbit at 1° / 3° / 10° per frame | Moving objects: near / all |
|---|---:|---:|---:|
| Every pixel, every frame | 0.0217 / 0.0175 / 0.0170 | 0.0167 / 0.0190 / 0.0233 | 0.0385 / 0.0215 |
| + settled skip at 128 | 0.0217 / 0.0175 / 0.0176 | (same) | (same) |
| + settled skip at 64 (default) | 0.0217 / 0.0175 / 0.0175 | (same) | (same) |
| + settled skip at 32 | — / 0.0188 / 0.0188 | (same) | (same) |
| Quarter-rate tiles | 0.0256 / 0.0214 / 0.0181 | 0.0190 / 0.0224 / 0.0255 | 0.0399 / 0.0256 |
| *Formula* | *0.0270* | *0.0270* | *0.0411 / 0.0300* |

Frame time on the wall clock, p10 / p50, interleaved over 6 rounds, with the moving objects drawn but out of the AO:

| Setting | Still, after 600 frames | Orbiting at 1° per frame |
|---|---:|---:|
| Formula | 0.31 / 0.43 ms | 0.34 / 0.36 ms |
| Every pixel, every frame | 1.88 / 2.07 ms | 2.08 / 2.30 ms |
| Settled skip at 128 | **1.30** / 1.42 ms | 2.00 / 2.11 ms |
| Quarter-rate tiles | 1.46 / 1.68 ms | **1.65** / 1.84 ms |
| Both | **1.14** / 1.35 ms | **1.42** / 1.73 ms |

**Shipped:** the settled skip at 64 is on, because it costs nothing measurable in accuracy. Quarter-rate tiles are off but available as `aoDemo.traced.tileRate.value = 4`. They save about 0.43 ms at any camera speed but give up about a fifth of the gain over the formula in motion and near moving objects.

### True half resolution (deferred)

The forward version traced in the scene draw itself, at full resolution, and wrote its history through extra render targets plus a copy to the screen. The traced mode is now deferred, in three passes at the AO resolution (`aoDemo.setAOScale`, 0.5 by default):
1. **Geometry pass:** world position and view depth (float), normal (half float) and a moving-surface flag.
2. **AO pass (full screen):** traces 2 directions per pixel and accumulates into the reprojected history. The settled skip and the optional tiles work as before.
3. **Scene draw:** writes straight to the screen. It reads the AO with depth- and normal-aware upsampling: the four nearest AO texels weighted bilinearly, each used only if it saw the same surface (view depth within 5% + 2 cm, normals within about 25°). If none qualifies, the pixel shows the formula alone.

At scale 1 the deferred pipeline reproduces the forward version exactly (0.0262 / 0.0217 / 0.0175 at 1 / 16 / 64 frames). At scale 0.5 it is more accurate everywhere, because the upsampling averages several noisy AO texels:

| Main view, AO error | Still: 1 / 4 / 16 / 64 / 256 frames | Orbit at 1° / 3° / 10° per frame | Moving objects: near / all |
|---|---:|---:|---:|
| Forward, full resolution (previous) | 0.0262 / 0.0253 / 0.0217 / 0.0175 / 0.0175 | 0.0167 / 0.0188 / 0.0225 | 0.0386 / 0.0220 |
| **Deferred, half resolution** | **0.0250 / 0.0220 / 0.0178 / 0.0160 / 0.0160** | **0.0164 / 0.0182 / 0.0212** | **0.0382 / 0.0202** |
| *Formula* | *0.0270* | *0.0270* | *0.0411 / 0.0300* |

Frame time, wall clock, p10 / p50, 12 interleaved rounds:

| Mode | Settled still camera | Orbiting at 1° per frame |
|---|---:|---:|
| Formula | 0.33 / 0.34 ms | 0.34 / 0.41 ms |
| Forward, full resolution (previous) | 1.29 / 1.37 ms | 1.86 / 2.02 ms |
| **Deferred, half resolution** | **1.21 / 1.27 ms** | **1.25 / 1.35 ms** |

A noisier 3-way run put deferred at full resolution at 2.10 / 2.58 ms (still / orbiting): the full-resolution geometry pass costs more than it saves. The AO mask showed no blockiness or halos at outlines at half resolution.

### Soft visibility (traced-ao Thermite campaign, champion 2)

The traced-ao campaign optimizes a numpy model of the estimator, [accuracy/traced_ao.py](../accuracy/traced_ao.py), against [its contract](../thermite/traced-ao.contract.json):
- **Simulation:** 64 frames on a still camera with the viewer's accumulation rule, at 20,000 pixels of each of the six views.
- **Score:** error after 16 frames.
- **Hard limits:** at most 14.25 field reads per pixel per frame (the seed uses 14.0); no worse than the seed after 1 or 64 frames.

Champion 2 replaced the hard hit test with soft visibility per direction: the closest approach, `min over steps of clamp((d + e) / (4e), 0, 1)` with `e = min(0.025, 0.1 · t · (w·n))`. Rays march until `d < -e`, with steps of at least 7.2 cm, up from 6 cm. On the CPU model it scored 5.7% better on the 16-frame score, and 4–6% better on each held-out view.

| Main view, GPU (M1, 1280 × 720, half resolution) | Hard hit (before) | Soft visibility |
|---|---:|---:|
| Still: 1 / 4 / 16 / 64 / 256 frames | 0.0250 / 0.0220 / 0.0178 / 0.0160 / 0.0160 | **0.0249 / 0.0217 / 0.0172 / 0.0153 / 0.0153** |
| Orbit at 1° / 3° / 10° per frame | 0.0164 / 0.0182 / 0.0212 | **0.0162 / 0.0174 / 0.0206** |
| Moving objects (4 held-out poses): near / all | 0.0382 / 0.0202 | **0.0380 / 0.0196** |
| Frame, still / orbiting (p10, 12 interleaved rounds) | 1.11 / 1.21 ms | 1.11 / 1.24 ms (no change) |

The GPU gain at 16 frames (3.4%) is smaller than the CPU model's (5.7%), as expected: the half-resolution upsampling already averages some of the noise the soft visibility removes.
