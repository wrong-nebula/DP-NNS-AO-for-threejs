# Speed plan: faster bake and runtime at the same AO quality

Quality floor for every change: **overall MAE ≤ 0.0401** for SDF wall correction and **≤ 0.0421** for the original four-tap SDF, both against the 512-ray reference ([final-accuracy.json](final-accuracy.json)). Wall over-darkening must stay ≤ 0.0217. Contacts, walls and moving-object occlusion must not visibly change.

Changes fall into two classes, and the class decides how much validation a change needs:

- **Exact.** The AO image is provably identical. Such a change skips only work that cannot change the result, or changes stored values only where the shader cannot read a difference. Validate it with a bit-for-bit or pixel-for-pixel diff against the champion.
- **Approximate.** Values change. Validate it against the ray-traced reference with the MAE guards, plus a visual check of contacts, walls and movers.

Measured on an Apple M1 (8 threads) in this checkout, unless marked *estimate* or *model*.

---

## Implemented and measured (branch `speed-prototypes`, uncommitted)

The machine was shared with other heavy sessions during testing (load average 6–9 on 8 cores). Every timing comparison below therefore ran champion and candidate **interleaved in the same session**. Each GPU result was repeated, and only differences that recurred are reported. Absolute numbers are inflated by the load; the ratios are what count.

| Change | Before | After | Quality check |
|---|---:|---:|---|
| Mover bake: bake only type-6 meshes at load (the rest on first "Baked SDF for all movers") | 60 ms | 41 ms | atlas **bit-identical** for both meshes |
| + winding-number term only inside the mesh's box | 41 ms | **12 ms** (5× overall) | 22 of 524k voxels differ by 1 half-float step (≤ 2 mm), no sign flips |
| Runtime: 1 m world-cell mover mask + early loop exit, plus a `NO_DYN` shader variant used when no mover is in the AO | wall correction, 11 movers: 0.746 / 0.805 ms | **0.620 / 0.620 ms** (−17 to −23%) | AO masks at 8 frozen poses × 2 modes: 88 of 14.7M pixels differ by 1/255, all from the smaller atlas's slab coordinate (present with the lazy bake alone) |
| | original 4-tap, 11 movers: 0.713 / 0.730 ms | **0.579 / 0.574 ms** (−19 to −21%) | same |
| | wall correction, **16 movers**: 1.104 / 1.138 ms | **0.787 / 0.829 ms** (−27 to −29%) | same 88 pixels; cloned mesh fields match |
| | original 4-tap, 16 movers: 1.057 / 1.077 ms | **0.710 / 0.721 ms** (−33%) | same |
| | static only (movers off): 0.253 / 0.247 ms | 0.240 / 0.244 ms (parity) | **pixel-identical** in both modes |
| Static baker: parallel BVH build, build only the active search, per-voxel Lipschitz cull, **1 triangle per BVH leaf** | dense 330 ms | **263 ms** (−20%; earlier quiet run 340 → 231) | **bit-identical**, Archive and dense; `bench_bake.sh` guards pass |
| | Archive 121 ms | **93 ms (−23%)** | bit-identical |
| `bake_sdf --reach 1.075` | | small (~3% fewer queries) | `eval_ao.py`: **AO identical at all 57,216 reference pixels**, MAE 0.04211 / 0.04012 |

Bake times are wall time until a forced readback (`aoDemo.bakeSyncMs`). The page's original `bakeMs` stops at `gl.finish()`, which doesn't wait on ANGLE/Metal, and it reported 23 ms for the same bake. A GPU elapsed-time query over the bake returned nonsense (18.7 s), so it isn't used. Medians are over 6–7 page loads; the first, cold-compile load (~520 ms) shows up in both.

**Hand-run hill-climb round on the baker** (`bench_bake.sh`, interleaved, guards: bit-identical fields and MAE). Each line is the candidate vs the champion of the time:

| Candidate | Archive | Dense | Verdict |
|---|---:|---:|---|
| BVH leaf 2 | −6.8% | +0.7% | below the 3% bar on dense; superseded |
| leaf 2 + block-centre seed | −8.6% | +3.9% | rejected |
| **leaf 1** + block-centre seed | −20.6%, then −23.5% | +0.9%, +3.0% | confirmed |
| **leaf 1** alone | **−20.8%** | −0.6% | **accepted** (simpler; the seed adds nothing) |
| block-centre seed alone | −0.8% | +0.6% | rejected |
| leaf 8 (single run) | +28% distance time | — | rejected |

All candidates were bit-identical; leaf size changes only which nodes are visited.

Tried and **rejected** (exact, but not faster on M1):
- Iterating only set mask bits with an open-ended loop. Lacking `findLSB` (WebGL2 is GLSL ES 3.00), it used `log2(m&-m)`. Everything got slower, static-only included (0.25 → 0.41–0.45 ms), presumably because the compiler can't bound the loop. Bounded loops with an early `break` once no higher bits remain are what shipped.
- Per-tap sphere lower bounds for every mover: +15% mover time. Analytic movers are cheaper to evaluate than to bound. Kept only for type-6 meshes (it saves atlas fetches), where it is neutral on M1.

A diagnostic split of the original mover cost explains the outcome. **Building the per-pixel mover mask cost ~0.31 ms, evaluating movers ~0.26 ms.** The world-cell mask attacks the larger share.

Thermite campaign for the static bake: [../thermite/](../thermite/README.md). The contract and kit validate with Thermite's own schemas, and the kit passed a dry run in the rig's harness image. Starting it needs owner decisions (commit, token, model, LLM budget).

Tools added: [mover-bench.js](mover-bench.js) (interleaved GPU A/B harness, frozen poses, mask and atlas diffs through IndexedDB, `?extra=5` for 16 movers, `refMAE`; baseline page [index.orig.html](index.orig.html)), [reference_movers.py](reference_movers.py), [eval_ao.py](eval_ao.py) (static MAE guard, reproduces the published 0.0421 / 0.0401 within 2e-5), [mover_cull_model.py](mover_cull_model.py).

Pairs of numbers are two separate interleaved runs. Scaling from 11 to 16 movers, the mover share above static grows by 71% in the original (0.49 → 0.84 ms) and by 41% in the candidate (0.38 → 0.53 ms). An earlier build without the `NO_DYN` variant was 8% slower static-only (0.247 → 0.268 ms); separate static shaders removed that.

### Ray-traced reference with moving objects

[reference_movers.py](reference_movers.py) traces the same 512 rays per pixel as `reference.py`, with **all 11 movers at all 8 benchmark poses**. It uses the viewer's exact transforms and the meshes' own triangles, exported to [meshes/](meshes/).
- Boxes and the sphere use analytic ray tests.
- The capsule, cylinder, rounded box and torus are sphere-traced on their exact SDFs.
- The teapot and torus knot (10,432 triangles) go through [tools/trace_tris.cpp](../tools/trace_tris.cpp), a BVH any-hit tracer.
- Its static part reproduces `reference.npz` exactly. Pixels whose camera ray hits a mover show the mover and aren't scored.
- [export_reference_movers.py](export_reference_movers.py) writes it for `moverBench.refMAE(tag)`, which scores the GPU masks read back from the viewer.

| GPU-mask MAE vs ray-traced reference (8 poses pooled) | Original viewer | Candidate |
|---|---:|---:|
| Mover-affected pixels (52,190), wall correction | 0.09197 | **0.09197** |
| Mover-affected pixels, original 4-tap | 0.09197 | **0.09197** |
| All scored pixels (~440k), wall correction | 0.04981 | **0.04981** |

Every one of the 16 pose × mode cases matches to 5 decimals: per-pose affected MAE ranges 0.0808–0.1043 in both. The speed changes leave mover accuracy unchanged. On the same pixels, the static-only GPU mask scores 0.0431 against numpy shader emulation's 0.0416; that gap is rasterization and filtering, not quantization. GPU-mask MAE is a slightly stricter, end-to-end measure than the published emulated 0.0401.

**Quality finding, independent of these changes.** On mover-affected pixels, the viewer's mover AO is *less* accurate than no mover AO at all: MAE 0.097 vs 0.086 at t = 0. With mover AO on, the mean signed error is −0.067 (too dark); with it off, +0.074 (too light). The four-tap formula over-darkens next to movers. This is a lead for a quality campaign, not a speed one.


---

## 0. Where the time goes

| Stage | Measured | Notes |
|---|---:|---|
| Static bake, Archive, 1,908 triangles | 110–113 ms | distance pass 105 ms · sign 3.6 ms · BVH 0.3 ms |
| Static bake, dense, 488,448 triangles | 308–345 ms | **BVH build 121–129 ms, single-threaded** · distance 146–171 ms · sign 17 ms · load 11 ms · **unused grid build 11 ms** |
| Static AO at runtime | 0.159 ms vs 0.091 ms with AO off | **AO itself costs only ~0.07 ms.** Even free static AO saves 0.07 ms. |
| Movers at runtime | 0.35–0.41 → 1.33–1.39 ms | **~1 ms. This is where runtime time goes.** |
| Mover bake (GPU) | 60–200 ms | brute force: every voxel × every triangle, with an `atan` winding term per pair |

Facts from the field and the shader that the plan relies on:

- The shader reads static distances only below **1.075 m** (`1.12 − .045`). Above that, every tap's occlusion clamps to 0.
- Archive voxels measure 20.3 × 14.1 × 6.3 cm. The diagonal is **0.255 m**.
- 36% of voxels end at the clip value, but block culling skips only 23%. 9% of voxels are inside geometry, and none are deeper than 0.6 m, so interior skipping gains nothing on the Archive.
- Half-float storage error is ≤ 0.49 mm for |d| < 1.12. The unused 8-bit `PACKED` path would have a maximum error of 5.3 mm.
- Of the 18,476 mover triangles baked at load, **~10.5k belong to the two type-6 meshes** (teapot and torus knot). The other nine movers render with analytic SDFs by default. Their baked fields serve only the "Baked SDF for all movers" toggle.

**Mover culling model** ([mover_cull_model.py](mover_cull_model.py)): 57,216 benchmark pixels, 11 movers, 16 poses, CPU emulation of the shader's tests.

| Per pixel (static surfaces) | Current | Tighter, still exact |
|---|---:|---|
| Candidate sphere tests (loop iterations after the instance mask) | 3.76 | **0.54** with a 1 m world-cell mask |
| Movers evaluated | 0.294 | 0.254 with an oriented-box test (87%) |
| Tap evaluations (4 per evaluated mover) | 67,274 | 49% with a per-tap sphere test · 39% with a per-tap box test · **31%** if taps where the static distance is already ≤ the mover's lower bound are also skipped |

The instance mask is loose because the floor and wall instances are huge: every mover reaches the floor's AABB.

---

## 1. Prioritized list

Ranked by expected gain per unit of risk. "Exact" means validated by diff, not by MAE.

| # | Change | Target | Gain | Class | Status |
|---|---|---|---|---|---|
| 1 | **Bake only type-6 meshes at load**; bake the rest on the first "Baked SDF for all movers" toggle | mover bake, atlas memory | removes ~43% of baked triangles; atlas 5.8 → 1.0 MB | exact for the default view | *estimate* |
| 2 | **Skip the winding-number term outside the mesh AABB** in `bakeFragmentShader` | mover bake | the `atan` term is about half the per-triangle ALU, and ~98% of voxels lie outside the AABB | exact (proof in §2.2) | *estimate* |
| 3 | **Parallel BVH build** plus **build only the search in use** | static bake | dense 340 → 231 ms (**−32%**) | exact, bit-identical | **prototyped** |
| 4 | **World-cell mover mask** (3D `R16UI` texture, 1 m cells, refilled each frame on the CPU) replacing the per-vertex instance mask and the per-pixel 16-wide loop | runtime, scaling | 3.76 → 0.54 candidate tests per pixel | exact (conservative superset) | *model* |
| 5 | **Per-tap lower bounds** (sphere or oriented box) for each candidate mover: skip taps where the bound ≥ min(static d, R_k − .045) | runtime | 69% of mover tap evaluations removed | exact | *model* |
| 6 | **Iterate only set mask bits** (`findLSB`, `m &= m-1`) instead of `for i<MAX_DYN … continue` | runtime | fewer loop iterations and less dynamic uniform indexing | exact | *estimate* |
| 7 | Per-voxel Lipschitz cull and upper-bound seed from the block centre, plus `--reach` | static bake | Archive ~4–8% (inside run noise) | exact, bit-identical; `--reach` is exact for AO | **prototyped** |
| 8 | **Ship the static field as half floats**, or cache it, instead of converting 2.1M floats in JS with `T.DataUtils.toHalfFloat` | page load | likely tens of ms *(unmeasured)* | exact (same bits) | idea |
| 9 | **Cache mover fields** (IndexedDB keyed on a geometry hash) or ship them prebaked | mover bake | 60–200 ms → ~0 ms after the first load | exact | idea |
| 10 | float32 + SIMD (4-wide BVH, structure-of-arrays triangle tests) in `bake_sdf` | static bake | *estimate* 1.5–2.5× on the distance pass | tiny value changes; validate by diff | idea |
| 11 | Analytic box path for `--boxes` input (SDF per box, BVH over 159 boxes instead of 1,908 triangles) | static bake | *estimate* several times faster | identical outside overlaps (the shipped field is analytic already) | idea |
| 12 | Two-level mover fields: a fine grid near the mesh, a coarse grid for the far pad | mover bake, atlas | *estimate* 4–8× fewer voxels | **approximate** | needs validation |
| 13 | 8-bit `PACKED` static field | bandwidth, memory | 4 → 2 MB | **approximate** (≤ 5.3 mm error) | needs validation |

---

## 2. Baker

### 2.1 CPU static baker (`tools/bake_sdf.cpp`)

**Prototyped on branch `speed-prototypes` (not committed):**

- **Parallel BVH build.** The top subtrees build on separate threads, with `2^depth ≤ threads`, and subtrees under 8,192 triangles build serially. The splits and the pre-order node layout are unchanged, so the node array is identical to the serial build. Dense BVH: 121–129 → 43–53 ms.
- **Build only the active search.** The uniform grid is built only for `--search grid`. Dense: −11 ms.
- **Per-voxel Lipschitz cull.** The block-centre query already gives the exact distance d_c. A voxel at offset r from the centre with `d_c − r > limit` is written without a query (118,600 queries skipped). `d_c + r` also seeds the BVH as an upper bound.
  - Pitfall found and fixed: adding even a tiny slack to the *seed triangle's* bound made Archive triangle tests rise 12%. Box scenes have many triangles at exactly equal distances, and strict pruning at the seed distance is what keeps them untested. Slack now applies only to the derived bound.
- **`--reach r`** sets the query limit to `r + voxel diagonal`, which is 1.330 m for r = 1.075. Values past the limit are written as `clip`.
  - Why this is exact for AO: a trilinear sample below r has all 8 corners within one diagonal of its minimum corner, so every corner it reads is exact. A sample above r can only rise, so it stays clamped to zero occlusion.
  - Verified: all 59,309 changed voxels have original values above the limit, and every one was raised.

Results (interleaved runs; this machine drifted ±20 ms between runs):

| | original | prototype | prototype + `--reach 1.075` |
|---|---:|---:|---:|
| Archive total | 110–126 ms | 104–118 ms | 103–122 ms |
| Archive triangle tests | 23.39 M | 22.50 M | 21.38 M |
| Dense total | 337–346 ms | **229–233 ms** | ~251 ms |
| Output | — | **bit-identical** (Archive and dense) | AO-identical by the argument above |

**Next steps, in order:**

1. **Cheaper queries.** The Archive is query-bound: 1.5M queries at about 0.5 µs per thread. Move to float32 with a 4-wide BVH (structure-of-arrays bounds) and test 4 triangles at a time. Output then differs at float rounding. Guard: max |Δ| below one half-float ulp at the value, so the uploaded half texture stays identical.
2. **Box fast path (`--boxes`).** Compute exact analytic box distance with a BVH over boxes. This matches the shipped analytic field bit for bit outside overlaps.
3. **Dense mesh.** Use a binned SAH build at more parallel depth (16 tasks to balance P and E cores), which also gives tighter queries. Replace the parser with `std::from_chars` (load is 11 ms).
4. **Riskier ideas:**
   - Closest-feature propagation: compute exact distances in a narrow band, then carry each voxel's nearest-triangle ID through sweeps. It is rarely inexact, so it needs a full-field diff.
   - Resolution matched to the anisotropic voxels, for example 128×88×40 (4.6× fewer voxels). This coarsens z, so contacts are at risk.
5. **Not DP-NNS.** precompute.md already shows that a sample-point seed costs more to build than it saves. The free previous-voxel seed is nearly as tight. Revisit only if a point index is already available for free.

### 2.2 GPU mover bake (`bakeFragmentShader` in `index.html`)

Today every voxel loops over every triangle: 262k voxels × 18,476 triangles ≈ 4.8 G pairs, each with a distance term and an `atan`.

1. **Lazy bake (exact).** Bake only `type===6` meshes at load. Bake the analytic shapes on the first `forceMesh` toggle. This also allocates only 2 slabs instead of 11.
2. **Winding only inside the mesh AABB (exact).**
   - From any point outside a convex set, the whole set lies in an open half-space of directions. The surface's solid angle is therefore < 2π, so |w| < 0.5, and the test `abs(winding)>.5` is already false there.
   - Guard: read back the atlas and diff against the champion. Expect bit-identical, which also guards against self-overlapping open meshes.
   - Gain: the `atan` term plus three `length`s for ~98% of voxels. Branches are coherent per tile.
3. **Far-voxel skip (exact for AO).**
   - First bake a 16³ coarse pass (1/64 of the work).
   - A fine voxel with `d_coarse(center) − |offset| > 1.075 + fine diagonal` is written as a large constant without the triangle loop. That only raises values, so the static argument above holds.
   - Expected ~40% of voxels: the 1.2 m pad dominates every cube.
4. **Per-triangle constants.** Precompute edges, normal, `1/dot2(edge)` and `1/dot2(nor)` once in JS into the triangle texture. This removes 4 divisions per pair.
5. **MRT.** Evaluate 4–8 z-layers per fragment so per-triangle setup is shared. Moderate gain.
6. **Cache or prebake (exact).**
   - Hash the position buffer and store the half-float slab in IndexedDB.
   - Alternatively, ship the demo's two meshes prebaked, which needs a generalized-winding sign option in the CPU tool for open meshes like the teapot.
7. **Time-slice** the bake across frames or into a worker so it never blocks a frame. Total cost is unchanged, but nothing stalls.
8. **Approximate, needs validation: two-level fields.**
   - A fine grid covers the mesh plus ~0.3 m. A coarse 16³ grid covers the 1.2 m pad.
   - Far from a mesh, distance is smooth. The interpolation error is about h²/(8R) ≈ 1 cm at 20 cm voxels and 0.5 m range, or ≈ 0.3% occlusion on the 0.8 m tap.
   - Gain: 4–8× fewer voxels and less atlas.
   - Guard: the mover reference in §4.

---

## 3. Runtime

The static AO adds only ~0.07 ms per frame. Fewer static taps or cheaper static sampling can't save much, and the 3-tap-like reweighting already cost MAE (0.0439). **Runtime work should go to movers.** Items 1–5 below are exact.

1. **World-cell mover mask.**
   - Each frame, the CPU marks the cells of a 26×18×8 grid (1 m cells, 3,744 texels, 7.5 KB as `R16UI`) that each mover's sphere, inflated by 1.12 m, overlaps.
   - The fragment shader does one `texelFetch` at `p` and then runs the exact sphere test only on set bits.
   - This replaces the vertex-stage `instMask` loop, which the floor and wall instances defeat. Model: 3.76 → 0.54 candidates per pixel.
   - It is the main **scaling** fix: cost then follows the movers near each pixel, not the mover count.
2. **Iterate set bits.** `while(m!=0){int i=findLSB(m); m&=m-1; …}` replaces the `for i<MAX_DYN` loops with `continue`.
3. **Per-tap masks from lower bounds.**
   - For each candidate mover, compute a 4-lane lower bound. Use the bounding sphere for everything, or the local box (same vec4 code as type 0) for the 9 boxed shapes.
   - Keep tap k only if `lb_k < min(d_static_k, R_k − .045)`. Below that threshold a mover can't lower `min` where it matters.
   - The mesh path then does only the needed `sdMesh` fetches. Analytic shapes skip the mover when no tap survives.
   - Model: 31% of tap evaluations remain.
   - Why the farthest-tap test is valid for any shape: distance is 1-Lipschitz, so `d(p+n·t) − t` never increases with t. That is the property the shader's comment states for spheres, and it holds in general.
4. **Oriented-box test at the farthest tap** in the per-pixel mask. It helps elongated movers (plank, pillar, capsule): 87% of movers survive.
5. **Uniform layout.** Upload `dynInv` as 3 `vec4` rows (mat3x4) and fold `dynType` into `dynParam`. Use a UBO so ANGLE/Metal reads less per mover.
6. **Depth prepass / overdraw.** The AO runs in the forward shader for every rasterized fragment.
   - On M1, tile-based hidden-surface removal already culls opaque overdraw, so expect ~0 there.
   - On desktop GPUs, a depth prepass with `depthFunc=Equal` is exact and can matter. Measure separately.
7. **Static path micro-optimizations.**
   - Pad the volume so out-of-range reads return +clip, which removes the per-tap bounds branches.
   - `PACKED` 8-bit (approximate) halves texture bandwidth.
   - Little to gain: ≤ 0.07 ms in total.
8. **Approximate, last: half-resolution mover term with bilateral upsampling, or temporal reuse.** Real risk to moving-object occlusion edges. Only try it if items 1–5 leave a gap.

---

## 4. Thermite campaigns for speed

Thermite runs hill climbs (push one metric while guards hold) and goal campaigns (work until a list of checks passes). The measurement harness doesn't fully exist yet, so the first campaign builds it.

### Campaign 0: goal campaign, measurement infrastructure

**Status: all four checks are built.**
1. [eval_ao.py](eval_ao.py)
2. [bench_bake.sh](bench_bake.sh): interleaved A/B, load average logged, bit-identity and MAE guards. Its last run: Archive −7.8%, dense −32.6%, GUARDS: PASS.
3. [mover-bench.js](mover-bench.js)
4. [reference_movers.py](reference_movers.py), with all 11 movers at 8 poses. Thermite itself isn't installed on the development machine, so no campaign has been run yet. These scripts are the metric and guard commands it would call.

All of these checks must pass before any speed campaign starts:

1. `eval_ao.py` emulates the shader in numpy at the 57,216 reference pixels from `reference.npz`, reading any `.f32` field. It reproduces 0.0421 and 0.0401 to 1e-4. Extend the static sampler in `mover_cull_model.py`.
2. `bench_bake.sh` runs the champion and the candidate **interleaved**, 7 rounds each, on the Archive and the dense mesh. It prints the medians and a `cmp` result. The interleaving matters: this machine drifted ±20% between consecutive runs.
3. A browser harness (`bench_movers.html`, or `aoDemo.benchmarkMovers()`) covers runtime:
   - A frozen list of 8 poses (`updateMovers(t)` at fixed t).
   - GPU timer-query medians over 25 rounds per mode.
   - `readPixels` of the AO mask (`showMask`) per pose.
   - Each slab layer of the mover atlas read back.
   - `bakeMs` over N reloads.
4. **Mover reference.** Extend `reference.py` with the 11 movers at the frozen poses:
   - Analytic ray tests for the boxes, sphere, capsule and cylinder.
   - Triangle ray tests for the rest.
   - Outputs: mover-region MAE (pixels within 1.12 m of a mover) and a contact band (pixels within 5 cm of a contact). This closes the gap that "accuracy with moving objects has not been measured" leaves today.

### Campaign A: hill climb, static bake time

- **Metric:** median `total` ms from `bake_sdf --stats`, from champion/candidate interleaved runs. Score = 0.5·Archive + 0.5·dense, relative to the champion.
- **Accept:** ≥ 3% better, confirmed by a second interleaved run.
- **Hard guards:**
  - The Archive and dense fields are bit-identical to the champion. Alternatively, for float changes, max |Δ| below a half-float ulp and the uploaded half texture identical.
  - Exception for `--reach`-style changes: identical wherever the value is ≤ the limit, and only raised elsewhere.
  - `eval_ao.py` MAE ≤ 0.0401 + 1e-4 (wall correction) and ≤ 0.0421 + 1e-4 (original).
  - The OBJ bake equals the box bake.
- **Search space:** §2.1 next steps 1–3. The prototype on `speed-prototypes` is the starting champion.

### Campaign B: hill climb, mover bake time

- **Metric:** median `aoDemo.bakeMs` over 7 reloads.
- **Hard guards:**
  - Atlas readback identical to the champion for type-6 slabs. Analytic slabs may be deferred, but must be identical once the toggle bakes them.
  - Default-view AO masks pixel-identical at all 8 poses.
- **Search space:** §2.2 items 1–7.
- Item 8 (two-level fields) is not allowed here. It goes to Campaign D.

### Campaign C: hill climb, runtime GPU ms with all 11 movers

- **Metric:** median GPU ms at 1280×720 over the 8 poses, for SDF wall correction with movers on.
- **Secondary metrics, logged:** static-only ms, and ms with movers duplicated to 16, to measure scaling.
- **Hard guards:**
  - AO masks at all poses are pixel-identical to the champion. Allow ≤ 1/255 on ≤ 0.01% of pixels, for driver FMA differences.
  - Static MAE unchanged.
- **Search space:** §3 items 1–6, in that order. Every item there is exact, so the guard is a strict diff. That is the right shape for a hill climb: no guard tuning, and no slow drift of MAE.

### Campaign D: hill climb, approximate changes under the quality floor

Use this only after A–C plateau.

- **Metric:** whichever of the A, B or C metrics the change targets.
- **Hard guards:**
  - Static MAE ≤ champion + 2e-4 (0.5%; the metric is deterministic, so any rise is real).
  - Wall over-darkening ≤ 0.0217 + 0.002.
  - Mover-region MAE ≤ champion + 2e-4.
  - Contact-band max error ≤ champion + 1/255.
  - No pixel worse than champion + 4/255.
- **Visual guard:** an automatic side-by-side crop of the stair contacts, the shelf/wall junction and each mover. Human sign-off is required before a change becomes champion.
- **Candidates:** §1 items 12–13 and §3 item 8.

---

## 5. Safe first vs careful validation

**Safe first:** exact, validated by diff, low effort.

1. Parallel BVH and skipping the grid. Done in the prototype.
2. Lazy bake of non-mesh movers.
3. Winding term only inside the mesh AABB.
4. Iterate set bits.
5. Ship or cache the static field as half floats, and cache mover fields.
6. Per-voxel Lipschitz cull (done) and `--reach` (done; exact for AO, but it changes stored values above the limit, so keep it opt-in).

**Exact, but measure before keeping:** these may not pay off on M1.

- World-cell mask, per-tap masks, oriented-box test, UBO layout.
- Depth prepass: likely zero gain on M1, possibly a real one on desktop.
- MRT and per-triangle constants in the GPU bake.
- float32/SIMD baker. Small value changes, so it needs the half-ulp guard.

**Needs careful validation:** approximate, Campaign D only.

- Two-level or coarse mover fields.
- 8-bit `PACKED` static field.
- Anisotropic or lower static resolution.
- Closest-feature propagation baking.
- Half-resolution or temporal mover AO.
- Fewer or reweighted static taps. Already shown to raise MAE: reweighting measured 0.0439.

---

## Reproduce

```bash
c++ -O3 -std=c++17 -pthread tools/bake_sdf.cpp -o bake_sdf        # on this Mac, add -isysroot <SDK> -I<SDK>/usr/include/c++/v1
./bake_sdf --boxes benchmark/archive.json --lo -1 -1 -1 --span 26 18 8 --res 128 --clip 1.5 --out a.f32
./bake_sdf --boxes benchmark/archive.json --lo -1 -1 -1 --span 26 18 8 --res 128 --clip 1.5 --subdiv 16 --out dense.f32
./bake_sdf ... --reach 1.075                                       # AO-exact far-space limit
python3 benchmark/mover_cull_model.py [field.f32]                  # culling headroom model
```
