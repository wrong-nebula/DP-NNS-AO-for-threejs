# Fast-SDF-AO Next-Gen Optimization & Realism Log

Branch: `feat/ao-next-gen` (off `main` `e600a61`)  
Hardware: Apple M1 (`ANGLE Metal Renderer: Apple M1`, WebGL2 + `EXT_disjoint_timer_query_webgl2`)

## Phase 0 & 1 — Baseline (`main` `e600a61`) & Judge v2

Established `benchmark/judge_v2.py` (real Apple M1 Metal GPU via Playwright Chrome):
- 6-view spatial, contact (5 cm), tail-risk (CVaR-5%), Oklab $L^*$, FLIP-style perceptual error, and GMSD edge error at frames `[1, 4, 16, 64]` + auto-cropped Top-5 worst-error board & FLIP heatmap PNGs.
- Camera-cut convergence curve (AUC, disocclusion frames 1..4), 32-frame 3°/frame orbit + settle, static flicker.
- Frozen & continuous-motion mover accuracy (4 held-out poses).
- Adversarial thin-plate (2/4/8/12/20 cm) + curved/organic (sphere + torus) exact triangle-traced fixtures.
- Real GPU timing (readback-closed wall + `EXT_disjoint_timer_query_webgl2`), startup latency, VRAM, and 7-mode zero-GL-error verification.

### Immutable Baseline Scorecard (`baseline-main`)

| Metric | `main` (`e600a61`) |
|---|---:|
| Fit MAE (1 / 4 / 16 / 64) | 0.02389 / 0.01684 / 0.00913 / 0.00746 |
| Held MAE (1 / 4 / 16 / 64) | 0.02209 / 0.01551 / 0.00836 / 0.00651 |
| Contact@16 MAE / underOcc | 0.02486 / 0.00910 |
| Contact@64 MAE / underOcc / overOcc | 0.02542 / 0.00601 / 0.01940 |
| Perceptual@64 Oklab / FLIP / FLIP-p95 / GMSD | 0.00365 / 0.02772 / 0.1036 / 0.02093 |
| Tail-risk CVaR-5%@64 | 0.05175 |
| Temporal ConvAUC / Disocc(1..4) / Flicker | 0.00841 / 0.01826 / 0.00080 |
| Orbit 3°/f end / Settle4 / Settle16 | 0.01150 / 0.01060 / 0.00836 |
| Diag `pix416854` (ref 1.000) / `pix575366` (ref 0.795) | 0.8314 / 0.4000 |
| Movers Frozen all/near · Cont all/near | 0.01185 / 0.03778 · 0.01188 / 0.03784 |
| ThinPlate meanMAE64 (20 cm pred vs ref) | 0.01090 (0.3933 vs 0.3411) |
| Curved (Sphere+Torus) MAE / maxErr | 0.05125 / 0.10200 |
| Wall ms/f (early / orbit / settled) | 3.106 / 2.200 / 0.912 |
| GPU ms/f (early / orbit / settled) | 5.863 / 4.006 / 1.478 |
| Startup contactBuild / VRAM / moverBake | 96.0 ms / 1429.1 KiB / 34.3 ms |
| GL errors (7 modes) / console errors | 0 / 0 |
