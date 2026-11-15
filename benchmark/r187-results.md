# Three.js r187 WebGPU comparison

[r187.html](r187.html) renders the same Archive scene, camera and 1280 × 720 resolution as [final.html](final.html), with Three.js r187 on its WebGPU backend. It sets up SSAO and GTAO the way the official `webgpu_postprocessing_ao` example does:
- an MRT pre-pass for packed view normals and velocity;
- the AO node at a resolution scale, with 16 samples and a 1.12 m radius, fed to the ambient light through `builtinAOContext`;
- GTAO temporally filtered and resolved with TRAA, and SSAO with its built-in blur;
- no MSAA in any mode.

The four-tap SDF is the viewer's "SDF wall correction" formula ported to TSL as the material's `aoNode`: no pre-pass and no post-processing pass.

Measured on an Apple M1 (Metal 3) in a Chromium-based browser, 2026-11-15.

## AO error against the 512-ray reference (57,216 pixels)

SSAO's intensity and GTAO's scale were swept on this scene and set to their lowest-error values: SSAO intensity 3 (swept 1–6) and GTAO scale 1.75 (swept 1–3). The SDF's wall weights were tuned on this scene too.

| Method | MAE |
|---|---:|
| **Four-tap SDF (wall correction)** | **0.0401** (matches the WebGL2 viewer's 0.0401) |
| GTAO 1× + TRAA | 0.0513 |
| GTAO 0.5× + TRAA | 0.0535 |
| SSAO 1× | 0.0627 |
| SSAO 0.5× | 0.0652 |
| *Legacy, for comparison: Three.js 0.170 SSAOPass / GTAOPass, full resolution* | *0.0792 / 0.0835* |

GTAO's error is the average of its AO target over 32 temporally rotated frames, which TRAA converges to on a still camera. SSAO is read after its blur. The SDF is read from a float render of its AO term.

## GPU time (Apple M1, shared and loaded machine)

WebGPU timestamp queries over 12 frames per sample, every mode interleaved, 40 rounds. Background load only ever adds time, so **p10** estimates the uncontended GPU cost, and the median shows the contended one.

| Mode | Frame p10 (p50) | AO cost over AO off (p10) | MAE |
|---|---:|---:|---:|
| AO off | 0.169 ms (0.273) | — | — |
| **Four-tap SDF** | **0.191 ms (0.295)** | **+0.02 ms** | **0.0401** |
| SSAO 0.5×, 16 samples | 0.257 ms (0.371) | +0.09 ms | 0.0652 |
| GTAO 0.5× + TRAA, 16 samples | 1.32 ms (1.88) | +1.15 ms | 0.0535 |
| SSAO 1×, 16 samples | 1.43 ms (2.10) | +1.26 ms | 0.0627 |
| GTAO 1× + TRAA, 16 samples | 4.35 ms (6.35) | +4.18 ms | 0.0513 |
| GTAO 1× + TRAA, **64 samples**, scale 1.25 | **6.74 ms (15.75)** | **+6.57 ms** | **0.0398** |

**Matching the SDF's accuracy.** SSAO cannot: its error stays at 0.063–0.065 from 16 to 64 samples, a floor set by what screen space can see. GTAO can, at full resolution with 64 samples and its scale retuned to 1.25: MAE 0.0398 against the SDF's 0.0401, reproduced to five decimals. At that setting a frame takes **6.74 ms against 0.19 ms, about 35× (p50: 53×)**. GTAO's AO work alone is about 6.6 ms against about 0.02 ms for the SDF's taps. At the r187 example's default (GTAO 0.5× + TRAA, 16 samples), GTAO is about 7× the SDF frame time, with 33% more error.

**Caveats.** One GPU (M1), one scene and camera, and a machine under heavy load from other work: medians spread up to ±50% between runs, so treat the ratios as an order of magnitude. Chromium rounds WebGPU timestamps (about 65 µs here), which matters little at these sizes. A first timing pass read every mode at about 0.2–0.3 ms, with SSAO below AO off; it was a measurement fault, and this table replaces it. On faster GPUs the gap shrinks: an outside report on other hardware measured GTAO 0.5× + TRAA at 0.458 ms against 0.209 ms for the SDF.

To repeat with unrounded timestamps, use Chrome with `--enable-dawn-features=allow_unsafe_apis --disable-dawn-features=timestamp_quantization` on an idle machine and run `await r187Bench.run()`.

## What this changes

The earlier headline ratios (SSAO 24× and GTAO 56× slower in the static benchmark; 6× and 15× with moving objects) compared the inline SDF against Three.js 0.170's legacy WebGL2 passes in their most expensive form:
- full resolution;
- a 32-sample SSAO kernel;
- each pass re-rendering the scene for its own normals and depth;
- full-resolution blur and denoise.

Against r187's current passes, the gap is smaller but still real on this M1. The SDF frame costs about the same as SSAO 0.5×, and about 7× less than GTAO 0.5× + TRAA. Matching its accuracy takes GTAO at full resolution with 64 samples, about 35× its frame time. SDF AO's advantages:
- **About 20–35% lower AO error than tuned GTAO + TRAA, and about 40% lower than tuned SSAO**, on this scene.
- **No pre-pass and no post-processing passes**, so nothing depends on screen resolution, TRAA history or ghosting.
- **Occlusion from geometry that is off-screen or hidden**, which screen-space methods cannot see.

Its costs: a precomputed field per scene (4 MB for the 128³ Archive volume, plus 0.5 MB per baked mesh mover), occlusion only inside the baked volume, and rigid moving objects only.
