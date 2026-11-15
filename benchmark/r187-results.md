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

## GPU time: no reliable split at this scale

Every mode measured roughly **0.2–0.3 ms per frame**, AO off included. That matches outside reports for r187 WebGPU on similar hardware: SDF 0.209 ms, SSAO 0.5× 0.237 ms, GTAO 0.5× + TRAA 0.458 ms.

The WebGPU timestamp queries could not separate the methods further on this setup:
- Chromium rounds them (here in steps of about 65 µs).
- They appear to miss some post-processing passes: SSAO 0.5× read *lower* than AO off.
- The machine was under heavy load (load average ~15).

For a precise split, run [r187.html](r187.html) in a browser with unrounded timestamps, for example Chrome with `--enable-dawn-features=allow_unsafe_apis --disable-dawn-features=timestamp_quantization`, on an idle machine: `await r187Bench.run()`.

## What this changes

The earlier headline ratios (SSAO 24× and GTAO 56× slower in the static benchmark; 6× and 15× with moving objects) compared the inline SDF against Three.js 0.170's legacy WebGL2 passes in their most expensive form:
- full resolution;
- a 32-sample SSAO kernel;
- each pass re-rendering the scene for its own normals and depth;
- full-resolution blur and denoise.

Against r187's current SSAO and GTAO, the costs are the same order of magnitude. SDF AO's remaining advantages:
- **About 20–35% lower AO error than tuned GTAO + TRAA, and about 40% lower than tuned SSAO**, on this scene.
- **No pre-pass and no post-processing passes**, so nothing depends on screen resolution, TRAA history or ghosting.
- **Occlusion from geometry that is off-screen or hidden**, which screen-space methods cannot see.

Its costs: a precomputed field per scene (4 MB for the 128³ Archive volume, plus 0.5 MB per baked mesh mover), occlusion only inside the baked volume, and rigid moving objects only.
