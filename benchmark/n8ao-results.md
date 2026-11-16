# N8AO 2.1 comparison (Three.js r187, WebGL2)

[n8ao.html](n8ao.html) renders the same Archive scene, camera and 1280 × 720 resolution as [final.html](final.html) and [r187.html](r187.html). N8AO 2.1 has no WebGPU version yet, so this page runs Three.js r187 on WebGL2.

The page compares four kinds of AO:
- **N8AO 2.1:** `N8AOPass` at the shared 1.12 m radius, in ten configurations (quality presets, half resolution, neural denoising, and denoising off).
- **The four-tap SDF:** both the r187.html formula and the viewer's current wall correction, inline in the forward shader.
- **Legacy passes:** r187's `SSAOPass` and `GTAOPass`, set up exactly as in final.html.

Raw numbers are in [n8ao-results.json](n8ao-results.json). Measured on an Apple M1 in Chromium (ANGLE on Metal), 2026-11-16, with the machine under background load (load average 15–30).

## AO error against the 512-ray reference (57,216 pixels)

N8AO is read from its "AO" display mode with gamma correction off. Its look was tuned the way SSAO's intensity and GTAO's scale were in r187.html: intensity (1–6) and distance falloff (0.25–2) were swept for the lowest converged error. The defaults won (intensity 3, falloff 1: 0.0427), ahead of 5 / 0.5 (0.0446).

Each N8AO configuration is scored two ways:
- **One frame:** the average error of single frames, noise included. This is what a viewer sees.
- **Converged:** the error after 96 frames of N8AO's own accumulation on the still camera. That removes the noise and leaves the method's floor.

| Method | One frame | Converged |
|---|---:|---:|
| **Four-tap SDF, current wall correction** | **0.0270** | (no noise: same) |
| Four-tap SDF, r187.html formula | 0.0401 | (same) |
| N8AO Ultra (64 samples) | 0.0410 | 0.0404 |
| N8AO Neural-Medium | 0.0424 | 0.0400 |
| N8AO Neural-Performance, half res | 0.0438 | 0.0437 |
| N8AO Medium | 0.0443 | 0.0427 |
| N8AO Neural-Performance | 0.0444 | 0.0420 |
| N8AO Neural-Medium, half res | 0.0445 | 0.0433 |
| N8AO Medium, half res | 0.0449 | 0.0439 |
| N8AO Performance, half res | 0.0452 | 0.0448 |
| N8AO Performance | 0.0462 | 0.0430 |
| N8AO 8 samples, denoising off | 0.0769 | 0.0394 |
| *r187 WebGPU, from [r187-results.md](r187-results.md): GTAO 1× + TRAA / SSAO 1×* | *0.0513 / 0.0627* | |
| Legacy SSAOPass / GTAOPass (r187) | 0.0792 / 0.0835 | |

**N8AO is the most accurate screen-space AO measured on this scene.** Every configuration scores 0.041–0.046, better than r187's GTAO + TRAA (0.051–0.054) and SSAO (0.063–0.065).
- **The SDF is still ahead.** Its current formula, at 0.0270, has a third less error than N8AO's best single frame, and less than N8AO's noise-free floor (about 0.040).
- **The denoiser costs accuracy.** With denoising off, N8AO's converged error is 0.0394. Its blur adds error back: 0.0430 with the Performance denoiser.
- **Neural denoising** improves single frames by 0.001–0.002 at the same sample count.

## Frame time

Frame time is measured on the wall clock: batches of 20 frames, each closed by a 1-pixel readback that waits for the GPU. That includes each method's CPU submission work. Rounds were interleaved across all modes, 30 rounds; background load only adds time, so p10 is the headline.

| Mode | Frame p10 (p50) | AO cost over AO off (p10) | One-frame error |
|---|---:|---:|---:|
| AO off | 0.25 ms (0.63) | — | — |
| **Four-tap SDF, current wall correction** | **0.32 ms (0.68)** | **+0.07 ms** | **0.0270** |
| Four-tap SDF, r187.html formula | 0.33 ms (0.64) | +0.08 ms | 0.0401 |
| N8AO Performance, half res | 2.09 ms (3.16) | +1.85 ms | 0.0452 |
| N8AO 8 samples, denoising off | 2.41 ms (3.09) | +2.16 ms | 0.0769 |
| N8AO Medium, half res | 2.84 ms (3.68) | +2.59 ms | 0.0449 |
| N8AO Performance | 3.11 ms (4.62) | +2.86 ms | 0.0462 |
| N8AO Neural-Performance, half res | 3.55 ms (4.54) | +3.30 ms | 0.0438 |
| N8AO Neural-Medium, half res | 4.29 ms (5.55) | +4.04 ms | 0.0445 |
| N8AO Neural-Performance | 4.93 ms (6.58) | +4.68 ms | 0.0444 |
| N8AO Medium | 5.34 ms (6.12) | +5.10 ms | 0.0443 |
| N8AO Neural-Medium | 8.82 ms (9.91) | +8.57 ms | 0.0424 |
| N8AO Ultra | 12.5 ms (14.7) | +12.3 ms | 0.0410 |
| Legacy SSAOPass (32 samples) | 2.63 ms (3.18) | +2.39 ms | 0.0792 |
| Legacy GTAOPass | 5.40 ms (6.56) | +5.16 ms | 0.0835 |

**What this means in time:**
- **The cheapest N8AO:** Performance at half resolution costs about 26× the SDF's AO time and about 6.5× its frame time, with 67% more error.
- **N8AO's most accurate single frame:** Ultra costs about 40× the SDF's frame time.
- **Denoiser cost:** Performance minus the same configuration with denoising off puts N8AO's spatial denoiser at about **0.7 ms** per full-resolution frame. Neural denoising adds another 1.8 ms at full resolution, or 1.5 ms at half resolution.

**The GPU timer is unreliable here.** Wrapped around the same frames, `EXT_disjoint_timer_query_webgl2` reported 1.4–3× more than the wall clock for multi-pass effects: N8AO Medium 13.4 ms against 5.3 ms, Ultra 36.5 ms against 12.5 ms, legacy GTAO 10.4 ms against 5.4 ms. A GPU timer cannot legitimately exceed the wall-clock time of the same frames. The likely cause is ANGLE's Metal backend splitting work while a timer query is active, which costs most in effects with many passes. The single-pass SDF read about the same either way.

Older WebGL timings in this repo used that timer, so their multi-pass figures are inflated. That covers final.html's legacy SSAO/GTAO numbers and the moving-object table in the README. The r187 WebGPU numbers use WebGPU timestamp queries and are not affected.

**Caveats:** one GPU (M1), one scene and camera, a loaded machine, and the browser page was not in the foreground. Compare ratios within this table rather than absolute times.

## Would a noisier, cheaper SDF plus N8AO's denoiser be faster?

No, not for speed. The whole four-tap SDF costs about 0.07 ms. N8AO's cheapest spatial denoiser alone costs about 0.7 ms at full resolution, ten times the entire SDF AO. Three things work against it:
- **A separate AO buffer.** Denoising needs the AO written to its own buffer and blurred in screen space. The SDF currently runs inline in the material shader with no extra passes.
- **Little to save.** A one- or two-tap SDF would save at most a few hundredths of a millisecond.
- **Blur adds error.** N8AO's own numbers show it: 0.0394 converged without denoising, 0.0430 with it.

The pattern does fit if more quality is wanted instead of more speed. True hemisphere integration from the SDF, with many directions each traced for occlusion, is too costly per frame at full sample counts. A few random directions per pixel, then a denoiser or temporal accumulation, would make it affordable. Whether that beats the fitted four-tap formula's 0.027 is an open question.
