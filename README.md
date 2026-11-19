# Fast SDF AO

Real-time ambient occlusion for Three.js from a baked signed distance field (SDF): a four-read formula, exact distances to moving objects, and an exact baker for meshes and point clouds. **Traced SDF (temporal)** combines exact near-surface triangle intersections (quantized SAH local BVH, or opt-in Hi-Z screen-space contact tracing) with an isotropic `256 × 176 × 80` FP16 SDF for farther visibility, half-resolution tangent-plane-aware upsampling, variance-guided temporal accumulation, and exact analytic/traced moving-object occlusion. See [nextgen-log.md](benchmark/nextgen-log.md) for current Apple M1 GPU accuracy, perceptual, temporal, and timing results (and [contact-results.md](benchmark/contact-results.md) for the earlier contact-repair baseline).

The project started from the DP-NNS nearest-neighbour paper, which motivates SDF baking, but it does not use DP-NNS: the baker computes exact distances to triangles with a BVH. *Why not DP-NNS* below explains why, with measurements.

This repository is an interactive Three.js ambient occlusion comparison using the Archive scene, with stairs, shelves, and walls, plus 11 moving objects. Orbit the scene, switch among seven AO modes, toggle **PBR + Multi-Bounce**, **Hi-Z contact**, **Smooth falloff**, or **AO mask**, and inspect the live FPS counter.

Serve this directory over HTTP and open `index.html` in a WebGL 2 browser. The scene and FP16 distance field are embedded; the contact accelerator loads from the local `contact-bvh.js` module. Three.js and its addons load from a pinned CDN URL.

## Traced SDF (temporal)

The **Traced SDF (temporal)** mode estimates the unblocked share of cosine-weighted directions out to 1.12 m. [contact-bvh.js](contact-bvh.js) supplies a generic triangle-mesh contact accelerator, including indexed and instanced meshes with world transforms. The farther estimator samples the baked isotropic FP16 distance field (`256 × 176 × 80`, ~10.1 cm voxels).

- **Sampling and tracing:** each AO pixel traces 4 directions on frames 0–7, 3 on frames 8–15, and 2 afterward (dropping to 1 on low-variance pixels after frame 24), using Cranley-Patterson IGN spatial shifts with exact 2D Halton(2, 3) sequences (`vdc2` elevation, `vdc3` azimuth) for spatial anti-correlation and wrap-free temporal stratification. Every direction first checks exact triangle intersections from a 0.3 mm normal-offset origin over the near segment (`max(0.24 m, 0.12 m + half voxel diagonal)`).
- **Far visibility:** unblocked directions hand off directly at `s = contactRange` to the SDF march (up to 16 steps of at least 7.2 cm, early exit at `v < 0.01`). Soft visibility is `min over steps of clamp((d + e) / (4e), 0, 1)`, where `e = max(1e-6, min(0.025, 0.1 * t * (w·n)))`. An opt-in **Smooth falloff** toggle fades occlusion smoothly over `[0.72, 1.12] m`.
- **Quantized SAH contact BVH:** a 1 m world grid selects a local 8-bin SAH triangle BVH built via O(T) 2-pass CSR cell binning with cell-clipped triangle AABBs. Nodes pack into single-fetch `RGBA32UI` (16 B/node, 16-bit local-cell quantization, ~23 µm precision) with split-axis-ordered short-stack traversal and deduplicated triangle storage. For the Archive this takes **~16.5 ms** CPU startup and **398.6 KiB** VRAM (5.8× faster build and 3.6× smaller than the initial float32 contact BVH). An opt-in **Hi-Z contact** toggle (`aoDemo.setContactMode(1)`) replaces the BVH with a 2-level Hi-Z min-depth screen-space contact trace (zero BVH memory/build).
- **Accumulation, upsampling, and variance filtering:** both temporal reprojection and full-resolution upsampling validate neighbors with **view-space tangent-plane distance** reconstructed from `viewDepth` (zero extra render targets or fetches), eliminating offset-plane shadow bleeding. History stays at **12 B/px** (`RG32F` AO + depth; `RGBA8` octahedral normal + Welford online `sqrt(var)*4` + `frames/255`), enabling variance-guided early settling (`frames >= 24 && var < 0.00015`), a variance-weighted formula blend, a 4-neighbor cross-bilateral prefilter on young noisy histories (`frames < 6`), and zero full-res SDF reads on settled pixels (`frames >= 32 && var < 0.0006`).
- **Moving objects and PBR showcase:** in Traced mode, nearby movers use closed-form analytic sphere occlusion (`sphOcclusion`) plus 24-ray stratified cosine-weighted exact ray-box/SDF visibility per frame (zero temporal lag, cutting near-mover error by 46.7%). Enabling **PBR + Multi-Bounce** (`aoDemo.setPBR(true)`) lazily enables a 3rd `bentOut` MRT for temporally accumulated **bent normals**, driving Jimenez multi-bounce colored AO, directional sky/ground ambient, GGX sun specular with GTSO specular occlusion, and ACES tone mapping + sRGB transfer (`AO mask` always outputs pure linear `[0, 1]` AO).
- **WebGPU world-space hash cache prototype:** [benchmark/webgpu-hash-ao.html](benchmark/webgpu-hash-ao.html) implements a native WebGPU compute world-space spatial hash AO cache (distance-LOD cells, tangent-plane jitter, LRU eviction, 100% converged-cell skip): `0.01126` frame-1 camera-cut revisit MAE (2× lower disocclusion error than screen-space) at `0.619 ms/f` settled on Apple M1 Metal-3.

Validated on Apple M1 GPU (`benchmark/judge_v2.py`) across 6 views, 5 cm contacts, perceptual metrics (Oklab, FLIP-style, GMSD, CVaR-5%), camera cuts/orbits, frozen & continuous movers, thin plates, and curved geometry:

| Apple M1 GPU (1280 × 720, 640 × 360 AO) | Previous contact BVH (`e600a61`) | **Current `main`** | Change |
|---|---:|---:|---:|
| Fitting MAE (1 / 4 / 16 / 64 frames) | 0.02389 / 0.01684 / 0.00913 / 0.00746 | **0.02234 / 0.01325 / 0.00845 / 0.00585** | -6.5% / **-21.3%** / -7.4% / **-21.6%** |
| Held-out MAE (1 / 4 / 16 / 64 frames) | 0.02209 / 0.01551 / 0.00836 / 0.00651 | **0.01966 / 0.01156 / 0.00738 / 0.00497** | -11.0% / **-25.5%** / -11.7% / **-23.7%** |
| 5 cm contact MAE@64 (underOcc / overOcc) | 0.02542 (0.00601 / 0.01940) | **0.01877 (0.00470 / 0.01407)** | **-26.2%** |
| Perceptual@64: FLIP / GMSD / CVaR-5% | 0.02772 / 0.02093 / 0.05175 | **0.02193 / 0.01051 / 0.03340** | **-20.9% / -49.8% / -35.5%** |
| Orbit 3°/f / Movers near (continuous) | 0.01150 / 0.03784 | **0.00975 / 0.02024** | **-15.2% / -46.5%** |
| Wall ms/f (early / orbit / settled) | 3.106 / 2.200 / 0.912 | **2.988** / 2.450 / 1.100 | **-3.8% early** |
| Contact BVH build / VRAM | 96.0 ms / 1,429.1 KiB | **16.5 ms / 398.6 KiB** | **5.8× faster / 3.6× smaller** |

Inspect `aoDemo.contactResources` for textures, memory, and build metadata. After changing static triangles or instance transforms, call `aoDemo.rebuildStaticContact()` (or supply a mesh array and `{gridSize, range, bias}`). This disposes the previous accelerator and resets temporal history. It does **not** rebake the far SDF, which must also be updated when static geometry changes.

**Historical SDF-only results:** the following table predates the contact accelerator and uses two directions per frame with blend constant 8. Its timings and 43% improvement describe that earlier implementation.

| SDF wall correction formula → traced (temporal), Apple M1, 1280 × 720 | Formula | Traced |
|---|---:|---:|
| AO error, still camera: 1 / 16 / 64 / 256 frames | 0.0270 | **0.0249 / 0.0172 / 0.0153 / 0.0153** |
| AO error arriving after an orbit at 1° / 3° / 10° per frame | 0.0270 | **0.0162 / 0.0174 / 0.0206** |
| AO error with all 11 objects moving (held-out poses): all pixels / near objects | 0.0300 / 0.0411 | **0.0196 / 0.0380** |
| Frame time (wall clock, p10; settled still camera / orbiting) | 0.33 / 0.34 ms | 1.21 / 1.25 ms |

In that historical experiment, aggregate error stayed below the formula from the first frame after a cut and settled 43% lower. The parameters were fitted on three views and checked on three held out; see [traced-results.md](benchmark/traced-results.md). This is not a per-pixel guarantee.

### Half resolution

Each frame runs three passes:
1. **Geometry pass, half resolution:** renders the scene's world position and view depth (float), normal (half float), and a flag for moving surfaces.
2. **AO pass, half resolution:** a full-screen pass that traces, reprojects the history and accumulates. Its cost depends on pixel count, active directions, and local geometry traversal.
3. **Scene draw, full resolution:** draws straight to the screen and reads the AO with depth- and normal-aware upsampling. Each pixel weighs the four nearest half-resolution texels bilinearly, using each only if it saw the same surface (view depth within 5% + 2 cm, normals within about 25°). Where none qualifies, at a silhouette or a thin edge, it shows the formula alone. The formula blend and the per-frame moving-object term run here at full resolution.

The following **historical** resolution comparison used the earlier hard-hit estimator, before soft visibility and the contact accelerator. Half resolution averaged several noisy AO texels per screen pixel and improved accuracy as well as cost. `aoDemo.setAOScale(1)` selects full resolution:

| Apple M1, 1280 × 720 | Full resolution | **Half resolution (default)** |
|---|---:|---:|
| AO error, still camera: 1 / 16 / 256 frames | 0.0262 / 0.0217 / 0.0175 | **0.0250 / 0.0178 / 0.0160** |
| AO error arriving after an orbit at 1° / 3° / 10° per frame | 0.0167 / 0.0188 / 0.0225 | **0.0164 / 0.0182 / 0.0212** |
| AO error with 11 moving objects: near objects / all pixels | 0.0386 / 0.0220 | **0.0382 / 0.0202** |
| Frame time, settled still camera / orbiting | 1.29 / 1.86 ms (forward) | **1.21 / 1.25 ms** |

Full resolution timed as the earlier forward version, which did no extra geometry pass. Deferred at full resolution took about 2.1 / 2.6 ms, because a full-resolution geometry pass costs more than it saves. The AO mask at half resolution shows no blockiness or halos at outlines.

## Moving objects

The static scene keeps its baked distance volume. Moving objects are not baked into it. The shader evaluates their distances at each AO tap and takes the minimum with the static volume, so moving or rotating an object needs no rebake.

- **Analytic shapes** use exact distance functions: box, sphere, capsule, cylinder, rounded box, and torus.
- **Rigid meshes of any shape** get a 64³ distance field baked once at load on the GPU, in the mesh's local space. Distance is exact point-to-triangle distance. The sign comes from the generalized winding number, so meshes that are not fully closed, such as the teapot, still bake correctly. All fields share one 3D atlas. Each field extends 1.2 m beyond its mesh, past the AO's reach, so nothing outside it can occlude. The demo bakes 11 fields with 18,476 triangles in 60–200 ms.
- **Culling** keeps the cost near the objects. Each draw instance first skips objects that cannot reach its bounds. Each pixel then skips any object whose bounding sphere the farthest tap, at 1.12 m along the normal, cannot reach. If the farthest tap cannot reach it, no nearer tap can. Convex shapes also skip themselves, because their own taps always point away from them. These tests only skip objects that cannot change the result. Renders matched pixel for pixel with them on and off.

Use **Movers in SDF AO** to compare with and without moving-object occlusion. **Baked SDF for all movers** switches the analytic shapes to their baked fields. Against the exact distance functions, about 8% of the pixels that moving objects affect change, by about 1/255 on average and at most 7/255.

Limits: up to 16 moving objects, rigid transforms only, and about 5 cm voxels in the demo's baked fields. Deforming meshes would need per-frame rebaking or a capsule approximation.

### Timings with all 11 objects moving

These are median GPU elapsed times per draw on an Apple M1 at 1280 × 720 over 25 rounds. The objects moved between rounds. They were measured in a visible browser tab with the render loop paused. This setup reads slower than the static benchmark below, which saw 0.16 ms for SDF wall correction. The SSAO and GTAO rows are multi-pass effects, which this WebGL timer overstates by about 1.4–2× on M1 (see *Static benchmark*), so their ratios here are upper bounds. Here the SDF modes took 0.35–0.41 ms without moving objects in AO, with the objects still drawn. Compare rows within this table.

| Method | Median (ms) | p95 (ms) |
|---|---:|---:|
| AO off | 0.20 | 0.28 |
| SDF modes, no moving objects in AO | 0.35–0.41 | 0.42–0.49 |
| SDF modes, exact shapes | 1.33–1.36 | 1.54–1.61 |
| SDF modes, all baked fields | 1.33–1.39 | 1.51–1.75 |
| Three.js 0.170 SSAOPass (legacy, full resolution) | 8.68 | 10.57 |
| Three.js 0.170 GTAOPass (legacy, full resolution) | 20.72 | 24.38 |

Against these legacy Three.js 0.170 WebGL2 passes, SSAO and GTAO took about 6× and 15× as long as the SDF modes with all 11 objects moving. **These ratios overstate the gap to current Three.js.** The legacy passes run at full resolution, re-render the scene for their own normals and depth, and blur at full resolution. Three.js r187's WebGPU SSAO and GTAO (half resolution, MRT normals, TRAA) narrow the gap: on an M1, SSAO 0.5× costs about the same as the SDF, and GTAO 0.5× + TRAA about 7× as much; see [r187-results.md](benchmark/r187-results.md). SDF cost grows with the number of moving objects near each pixel. SSAO and GTAO cost the same regardless of what moves. Accuracy with moving objects is measured against a ray-traced reference that includes all 11 objects at 8 poses ([reference_movers.py](benchmark/reference_movers.py)); see *Moving-object AO accuracy* below.

## Static benchmark

The [benchmark](benchmark/final-benchmark.md) compares four-tap SDF AO with the legacy Three.js 0.170 SSAO and GTAO passes on an Apple M1 in the static scene. A [comparison with Three.js r187's WebGPU SSAO and GTAO](benchmark/r187-results.md) finds the SDF about as fast as SSAO at 0.5× and about 7× faster than GTAO 0.5× + TRAA on an M1, with lower AO error: 0.040, against 0.051–0.053 for tuned GTAO and 0.063–0.065 for tuned SSAO. Reaching the SDF's accuracy takes GTAO at full resolution with 64 samples, about 35× the SDF's frame time; SSAO does not reach it at any sample count. It includes [GPU timings](benchmark/final-timings.json), [accuracy measurements](benchmark/final-accuracy.json), and a [side-by-side comparison](benchmark/final-comparison.png) against ray-traced AO. The live FPS counter measures browser frame cadence. The benchmark uses GPU elapsed queries.

The **historical SDF-only traced mode** settled at error 0.015 and about 1.2 ms per frame in that experiment. Those timings do not include the current contact accelerator; use [contact-results.md](benchmark/contact-results.md) for the current renderer. The historical GTAO comparison used a different backend (WebGPU versus WebGL2).

[N8AO 2.1](benchmark/n8ao-results.md), the most accurate screen-space AO measured here, was added on Three.js r187 (WebGL2, since N8AO has no WebGPU version yet):
- **Accuracy:** its configurations score 0.041–0.046 per frame, better than r187's GTAO and SSAO. Its noise-free floor is about 0.040. The SDF's current formula scores 0.027.
- **Speed:** its cheapest configuration (Performance, half resolution) takes about 2.1 ms a frame against 0.32 ms for the SDF; its AO work costs about 26× the SDF's. Its spatial denoiser alone costs about 0.7 ms, ten times the SDF's whole AO.

**Timer caveat:** on Apple M1 / ANGLE Metal, the WebGL `EXT_disjoint_timer_query_webgl2` timer read 1.4–3× too high for multi-pass effects, compared with wall-clock time closed by a readback. N8AO Medium read 13.4 ms against 5.3 ms, and the legacy GTAOPass 10.4 ms against 5.4 ms. The legacy SSAO/GTAO figures measured with that timer are inflated by the same effect, both in this section and in the moving-object table above. Measured on wall-clock time, r187's legacy SSAOPass and GTAOPass take 2.6 ms and 5.4 ms a frame against the SDF's 0.32 ms: about 8× and 17×, not 24× and 56×. The r187 WebGPU comparison uses WebGPU timestamp queries and is not affected.

The Archive volume was baked from analytic box distances. The runtime AO shader samples the baked volume.

## Baking a field for your own scene

[`tools/bake_sdf.cpp`](tools/bake_sdf.cpp) bakes a clipped signed distance field (cubic `--res 128`, anisotropic/isotropic `--res-xyz 256 176 80`, direct binary16 `--f16`, or 2-level sparse narrow-band `8³` brick-map `--brick-out`) from any closed triangle mesh (OBJ), box list (JSON) or oriented point cloud (PLY). It has no dependencies:

```
c++ -O3 -std=c++17 -pthread tools/bake_sdf.cpp -o bake_sdf
./bake_sdf --obj scene.obj --lo -1 -1 -1 --span 26 18 8 --res-xyz 256 176 80 --clip 1.5 --f16 --out scene.f16
```

Distances are exact (closest point on the nearest triangle), far space is skipped, sign comes from winding numbers, and all cores are used. On an Apple M1, the Archive takes about 35 ms at `128³` (~51 ms at isotropic `256 × 176 × 80` FP16) and a 488k-triangle version of it about 150 ms (originally 115 ms and 307 ms; see below). Outside the geometry the output matches the analytic field exactly, and AO accuracy is unchanged. See [precompute.md](benchmark/precompute.md).

### From a point cloud (scans, photogrammetry, splats)

```
./bake_sdf --points scan.ply --lo -1 -1 -1 --span 26 18 8 --res 128 --clip 1.5 --out scan.f32
```

The input is a PLY vertex list, ASCII or binary little-endian, with `x y z` and optionally `nx ny nz`; other properties are ignored.
- **Distance:** to the nearest point, found exactly with a BVH over the points (median splits, four points per leaf, built in parallel).
- **Sign:** from the side of that point's tangent plane. Clouds without normals, such as 3D Gaussian splat exports whose normals are all zero, bake an unsigned field. AO reads the field outside surfaces, where unsigned and signed agree, but thin open surfaces can't be told apart from solids.
- **Culling:** the same Lipschitz culling as meshes, since distance to a point set is 1-Lipschitz too. A culled voxel takes its reference point's sign: the segment between them stays farther than the clip distance from every point, so it crosses no sampled surface.

`--sample N` (with `--boxes` or `--obj`) bakes from N area-uniform samples of the mesh instead. Samples on surfaces covered by other geometry, such as the underside of a box standing on the floor, are dropped, as a scanner would not see them; a point a hair outside each sample along its normal is tested against the mesh's winding number. `--write-points f.ply` saves the samples.

| Archive, 1M samples requested (852k kept, 148k hidden) | Points | Exact triangles |
|---|---:|---:|
| Mean / p95 distance error within AO reach | 8.5 / 19 mm | 0 |
| Voxels near surfaces with the wrong sign (at contacts) | 0.14% | 0 |
| AO error vs ray-traced, original four-tap | 0.0428 | 0.0421 |

About 2.5M samples bring the AO error within 1% of the exact field's (see *Why not DP-NNS*). With a mesh, bake from the mesh: it is exact and about 20–30× faster.

### Exactness check

`--validate K` compares K random voxels with brute force after the bake: the distance to every triangle and the winding number of every triangle along the voxel's column (or every point, for point clouds). Any mismatch exits with status 1. [bench_bake.sh](benchmark/bench_bake.sh) runs it on the Archive (50,000 voxels) and the dense mesh (1,000) for every candidate baker. Unlike the bit-identity guard, which only compares with the previous baker, this catches a change that is wrong in a consistent way: a deliberately unsafe culling bound failed it on 1,197 of 20,000 voxels.

## Why not DP-NNS

[DP-NNS](https://arxiv.org/abs/2611.15320) answers exact nearest-neighbour queries over a point set. It records, at each insertion into a Delaunay triangulation, which earlier points the new point became a neighbour of, then walks that graph from a good starting point. A faithful implementation (CGAL Delaunay with exact predicates, the paper's level-consistent parallel build and grid starting points, Morton-ordered queries) was benchmarked against a kd-tree (nanoflann) and the point BVH above, on the 2.1M voxel queries of a 128³ Archive bake. All three return the exact nearest point for every query, so they produce identical fields and only their speed differs. Full results and code: [research/dpnns/RESULTS.md](research/dpnns/RESULTS.md).

| 851k points, 8 threads | Build | 2.1M queries | Total |
|---|---:|---:|---:|
| kd-tree (nanoflann) | 50 ms | 1,137 ms | 1,187 ms |
| BVH over points | 113 ms | 718 ms | **831 ms** |
| DP-NNS (64³ grid start) | 3,349 ms | 229 ms | 3,578 ms |
| *Exact triangle bake, for comparison* | | | *30 ms* |

- **Fastest queries, slowest build.** DP-NNS queries are 2.7–5× faster than the kd-tree's and 1.8–4.6× faster than the BVH's, as the paper reports. Its Delaunay build is 18–50× slower than the BVH's, and a 128³ bake doesn't have enough queries to pay that back: break-even is about 8–14M queries on one cloud, a 200³–240³ grid.
- **Not for meshes.** Turning a mesh into points first loses exactness (8.5 mm mean error at 852k points, wrong signs at contacts), and even the fastest point bake is about 20–30× slower than the exact triangle bake.
- **Not for real time.** It runs on the CPU, each query walks hundreds of graph edges whose count varies per pixel (a poor fit for GPUs), and moving any point means rebuilding the graph. The shader instead reads the baked field with hardware trilinear filtering. Moving objects use exact analytic distances or rigidly moved baked fields.
- **Not for this repository's licence.** CGAL's 3D triangulation package is GPL or commercially licensed.

DP-NNS fits many queries against one fixed point cloud: the paper's ICP registration, or AO textures of 2048² and up baked directly from a dense scan.

## What I improved on the first version


This section records historical improvements before the current triangle-contact renderer. The speed changes below kept the then-current AO unchanged; the accuracy changes were measured against ray-traced AO:
- **Static bake:** baked fields are bit-identical to the original wherever the AO can read them.
- **Static AO:** MAE stays 0.0401 (wall correction) / 0.0421 (original four-tap) against the 512-ray reference.
- **Moving objects:** AO error is unchanged against a new ray-traced reference that includes all 11 moving objects at 8 poses ([reference_movers.py](benchmark/reference_movers.py)).

Measured on an Apple M1.

| What | Before | After |
|---|---:|---:|
| Static bake, Archive (1,908 triangles), 8 threads | ~115 ms | **~35 ms** (3.3× faster) |
| Static bake, dense Archive (488k triangles), 8 threads | ~308 ms | **~150 ms** (2× faster) |
| Moving-object bake at page load | 60 ms | **12 ms** (5× faster) |
| Runtime AO, 11 moving objects | 0.79 ms | **0.62 ms** (−21%) |
| Runtime AO, 16 moving objects | 1.10 ms | **0.79 ms** (−28%) |
| Static AO error, SDF wall correction (vs ray-traced, GPU output) | 0.0408 | **0.0275** (−33%) |
| AO error near moving objects (vs ray-traced, held-out poses) | 0.097 | **0.041** (−58%) |
| Static AO error, new Traced SDF (temporal) mode at half resolution, still camera (vs SDF wall correction) | 0.0270 | **0.0153** (−43%) |

### Static bake (`tools/bake_sdf.cpp`)

The bake is dominated by about 1.5 million exact closest-triangle queries, one per voxel within reach of a surface. Most of the work went into making each query cheaper and skipping queries that cannot change the output. The BVH's node layout and traversal order never affect which triangle is nearest, so every change below produces a bit-identical field.

- **Binned SAH build.** The BVH used to split each node at the median centroid along its longest axis. It now picks the split from 16 bins per axis by the surface-area heuristic, down to one triangle per leaf. Node visits per query fall from 35.8 to 10.0 on the Archive and from 39.6 to 17.1 on the dense mesh. Leaf bounds are stored in float32, rounded outward, so every box still contains its triangles and pruning stays conservative.
- **Cache-line nodes.** After the build, internal nodes are packed into 64-byte nodes that hold both children's bounds (structure-of-arrays by axis) and their references. One-triangle leaves become direct references in their parent. Traversal loads one cache line per visit, tests both children's box distances together with NEON in double precision, and keeps each child's distance on the stack so a popped node is never re-tested. A stack of `{int, double}` pairs defeated store-to-load forwarding and was 1.6× slower than two separate arrays.
- **In-place, parallel build.** The SAH build writes nodes straight into their final pre-order slots, partitioning in place (no intermediate tree), with the top subtrees built on separate threads. The winding-number sign pass starts on its own thread right after loading and overlaps the tree build and the distance pass. Distances are written unsigned, and inside voxels are negated at the end.
- **Hierarchical Lipschitz culling.** A signed distance field is 1-Lipschitz: it changes by at most the distance moved. Each 8³ block queries its centre once. Sub-blocks of 4³, then 2³, whose bound already puts every voxel beyond the clip distance are filled without per-voxel queries, and the remaining voxels start their search from the centre's bound. The bounds come from a precomputed half-voxel distance table, so the culls need no square roots.
- **Cheaper sign pass.** The sign comes from counting surface crossings along a vertical ray per voxel column. Triangles whose three vertices share an x or y coordinate project to a line and can never be crossed, so they are skipped. The rest are binned per column in a compact (CSR) layout.
- **Earlier exact steps:**
  - the top levels of the tree build run in parallel and produce the same node array;
  - the unused uniform-grid search is no longer built;
  - each voxel's lower bound from its block centre skips queries that cannot land inside the clip distance.
- **Optional `--reach r`:** values beyond the AO's reach plus one voxel diagonal are written as the clip value without being resolved. A trilinear sample below the reach reads only corners within one diagonal of each other, so the AO is identical.

### Moving-object bake (GPU, at page load)

Each mesh's 64³ field is baked by a fragment shader that loops over every triangle for every voxel, computing the exact distance and a generalised winding number (sum of solid angles) for the sign.

- **Lazy baking.** Only the meshes that render through their baked field (the teapot and torus knot, 10.4k of 18.5k triangles) are baked at load. The analytic shapes are baked the first time "Baked SDF for all movers" is switched on, and the atlas shrinks from 11 slabs to 2. 60 → 41 ms.
- **Winding term only inside the mesh box.** From any point outside a convex set that contains the mesh, every triangle lies within an open half-space of directions. The total solid angle is then below 2π and the winding number below ½, so the sign is always "outside". The `atan`-based winding term, about half the cost per triangle, is evaluated only for voxels inside the mesh's bounding box, a few percent of the 3.3 m cube. 41 → 12 ms.

### Moving-object AO accuracy

Moving objects entered the AO through `min()`: each tap took the nearer of the static field and the nearest object, so an object's occlusion counted as fully as a wall's at the same distance. A small object blocks much less of the hemisphere than a wall, and against the ray-traced reference the pixels near moving objects came out too dark: on average by 0.07, with an error of 0.097. That is *worse* than leaving the objects out of the AO, which makes them too light (0.093).

The shader now keeps the static-only tap distances and weights only the extra occlusion the objects add: `occ = occStatic + (occAll − occStatic) · moverScale`. The weight 0.5 was fitted on 4 of the reference's 8 poses and confirmed on the other 4, in both SDF modes:

| Held-out poses | Before (weight 1) | Objects ignored (weight 0) | **After (weight 0.5)** |
|---|---:|---:|---:|
| Error near moving objects | 0.097 | 0.093 | **0.050** |
| Mean bias near moving objects | −0.070 (too dark) | +0.085 (too light) | **+0.007** |
| Error over all pixels, wall correction / original | 0.050 / 0.052 | 0.048 / — | **0.043 / 0.045** |

Per-tap weights gained only 0.001 more, with an irregular pattern, so the single weight was kept. Pixels without a moving object in reach, and every pixel with moving objects off, are unchanged: the static masks are pixel-identical in all three modes. The cost is a few vector operations and no extra texture reads. `aoDemo.setMoverScale(w)` changes the weight live, and `moverBench.scoreMoverScale([...])` re-runs the fit.

### Static AO accuracy (SDF wall correction)

The wall-correction formula was refit against ray-traced AO, with the same four texture reads per pixel. The formula is one function, [accuracy/ao_formula.py](accuracy/ao_formula.py), ported line for line to `aoWallCorrection()` in the shader. It was scored on six camera views of the Archive; three of them were held out, and a change that made any of those worse was refused.
- **Per-tap response curves.** Away from the structural walls, each tap has its own bias and a cheap curve: the near tap squared, the middle two square-rooted, and the far tap to the 8th power (three squarings), so only deep occlusion counts.
- **Tilted taps on vertical faces.** On faces that are not horizontal, the fourth tap sits 0.8 m out at 45° up instead of 1.12 m along the normal, to see overhangs such as shelf boards and the stair underside. The third tap sits 0.7 m out and 0.4 m down, to see the floor and supports below (wall–floor and crate–floor corners). Floors and tops keep both along the normal, and moving objects are tested at the same positions.
- **Separate fits per surface class.** Floors and tops, other faces, and the two structural walls each have their own weights. The walls use only linear clamps and one square, with each clamp's kink away from where a wall's taps sit.
- **Scored as the GPU renders.** The CPU evaluator models the GPU's 8-bit trilinear weights and half-precision filtering. Without that, a wall refit using `sqrt` of a value that is about 0 on every wall scored 8% better in full precision, yet darkened walls by 0.013 on the GPU.

| SDF wall correction, GPU output vs ray-traced | Before | After |
|---|---:|---:|
| All pixels | 0.0408 | **0.0275** |
| Structural walls | 0.0212 | **0.0141** (still slightly lighter than the reference, so no extra darkening) |

The "Original SDF" and "Reweighted SDF" modes are unchanged, pixel for pixel. With moving objects, all-pixel error is 0.030, from 0.050.

### Runtime AO with moving objects

Splitting the original cost showed that deciding which objects can affect each pixel (about 0.31 ms) cost more than evaluating them (about 0.26 ms). Every floor and wall pixel looped over all 11 objects, because the large floor and wall instances are within reach of every object.

- **World-cell object mask.** Each frame the CPU fills a 26 × 18 × 8 grid of 1 m cells (an `R32UI` 3D texture, 7.5 KB), setting object *i*'s bit in every cell within its bounding radius + 1.12 + (1.12 − 0.044) m. That set contains every pixel the exact per-pixel test can pass. The fragment shader reads one texel, ANDs it with the per-instance mask, and runs the exact sphere test only on the bits left. Candidate tests per pixel fell from 3.8 to about 0.5.
- **Bounded loops with early exit.** The object loops keep a fixed bound and `break` once no higher bits are set. Open-ended loops over set bits (`m &= m - 1`) made every mode slower, the static path included, presumably because the compiler can no longer bound them.
- **Separate no-object shader.** When no moving object is in the AO, a `NO_DYN` variant without any object code is used, so the static path pays nothing for the object support.

Measurements and rejected ideas: [speed-plan.md](benchmark/speed-plan.md).

References: [DP-NNS paper](https://arxiv.org/abs/2611.15320)
