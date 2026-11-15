# Final Archive AO benchmark

The Apple M1 rendered the same 159-box Archive scene at 1280 × 720 with a 1.12 m nominal radius for every AO mode. GPU elapsed times are per draw within 20 rotating batches of 12 draws. Times include scene rendering and exclude SDF baking and texture upload. SSAO and GTAO use the Three.js 0.170.0 passes with full-resolution depth and normal rendering and filtering. An engine with reusable depth and normal buffers may have different ratios, and Three.js r187's WebGPU passes do: see [r187-results.md](r187-results.md), where SSAO and GTAO cost about the same as the SDF AO.

AO error is mean absolute difference on the 0–1 visibility scale over 57,216 visible sample points, against a matched reference with 512 cosine-weighted rays per point. The wall-specific SDF weights were tuned on Archive. Figures describe these configurations on one camera and scene.

| Method | GPU median (ms) | GPU p95 (ms) | Overall AO MAE |
|---|---:|---:|---:|
| AO off | 0.091 | 0.153 | — |
| Original four-tap SDF | 0.161 | 0.308 | 0.0421 |
| Reweighted SDF | 0.139 | 0.214 | 0.0439 |
| SDF wall correction | 0.159 | 0.275 | 0.0401 |
| Three.js SSAO | 3.797 | 4.205 | 0.0792 |
| Three.js GTAO | 8.970 | 9.488 | 0.0835 |

To repeat, serve the outputs directory with `python3 -m http.server 8000` and open `http://localhost:8000/benchmark/final.html`. Run `await aoDemo.benchmark()` in the browser console.
