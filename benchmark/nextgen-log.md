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

---

## Phase 2 — Tangent-Plane Rejection, Halton(2,3)+IGN Sampling, Far-March Handoff & Variance-Guided Filtering

### Changes Implemented
1. **Zero-Bandwidth View-Space Tangent-Plane Distance Rejection** (`index.html`):
   - Reconstructs exact view-space positions of half-res neighbors directly from `viewDepth` (`invProjXY * h.y, -h.y`) with zero extra render targets or texture fetches.
   - Replaces raw depth thresholds in both `tracedAt()` and `history()` with `abs(dot(vn, vpQ - vp)) <= 0.0015 * viewDepth + 0.005` + grazing-friendly depth guard `0.08 * viewDepth + 0.05`.
   - Eliminates offset-plane bleed (`pix416854`: `0.8314 → 1.0000` exact).
2. **Cranley-Patterson IGN + 2D Halton(2,3) Sampling** (`contact-bvh.js`):
   - Replaces the broken `g1/g2 + mod(frame,64)` IGN azimuth scrambling with per-pixel constant IGN spatial shifts in both `u` and `phi` + exact `vdc2(frame)` elevation stratification + `vdc3(4*frame+i)` azimuth sequence (zero wrap, 2×2 anti-correlated).
3. **Far SDF March Handoff** (`contact-bvh.js`):
   - Starts far SDF march at `s = contactRange` (eliminating redundant `[0.12, 0.2474]` overlap with exact BVH) + early exit `v < 0.01`.
   - *Ablated & rejected*: Aaltonen 2-sphere soft visibility regressed MAE to `0.224` because `d > prevD` on rays leaving surfaces triggers false `h = 0` occlusion.
4. **Variance Moments, Early Settling, Control-Variate Blend & Young-History Prefilter** (`index.html`):
   - Upgraded `histTex` from `RG32F` to `RGBA32F` `(aoMean, viewDepth, aoM2, variance)`.
   - Variance-guided early settling (`frames >= 24 && var < 0.00015`) and adaptive ray budget (`1` ray for low-variance pixels after frame 24).
   - Variance-weighted Bayesian/control-variate blend `effK = clamp(blendK * (0.55 + 8 * var), 1.5, 4.5)`.
   - Lightweight 4-neighbor cross-bilateral prefilter applied *only* to young noisy histories (`frames < 6 && var > 0.003`).

### Phase 2 Results (`p2-final` vs `baseline-main`)

| Metric | `baseline-main` | `p2-final` | Delta |
|---|---:|---:|---:|
| Fit MAE (1 / 4 / 16 / 64) | 0.02389 / 0.01684 / 0.00913 / 0.00746 | 0.02200 / 0.01347 / 0.00931 / 0.00718 | -7.9% / **-20.0%** / +2.0% / -3.8% |
| Held MAE (1 / 4 / 16 / 64) | 0.02209 / 0.01551 / 0.00836 / 0.00651 | 0.01992 / 0.01177 / 0.00818 / 0.00628 | -9.8% / **-24.1%** / -2.2% / -3.5% |
| Contact@64 MAE / underOcc / overOcc | 0.02542 / 0.00601 / 0.01940 | 0.01936 / 0.00601 / 0.01335 | **-23.8%** / 0.0% / **-31.2%** |
| Perceptual@64 Oklab / FLIP / p95 / GMSD | 0.00365 / 0.02772 / 0.1036 / 0.02093 | 0.00335 / 0.02624 / 0.0932 / 0.01243 | -8.2% / -5.3% / **-10.0%** / **-40.6%** |
| CVaR-5%@64 | 0.05175 | 0.04337 | **-16.2%** |
| Temporal ConvAUC / Disocc(1..4) / Orbit3° | 0.00841 / 0.01826 / 0.01150 | 0.00824 / 0.01564 / 0.00997 | -2.0% / **-14.3%** / **-13.3%** |
| Diag `pix416854` (ref 1.000) | 0.8314 | **1.0000** | Fixed |
| ThinPlate meanMAE64 | 0.01090 | 0.00612 | **-43.9%** |
| Curved (Sphere+Torus) MAE | 0.05125 | 0.03958 | **-22.8%** |
| GL errors (7 modes) | 0 | 0 | Pass |
