# DP-NNS AO for Three.js

An interactive Three.js ambient occlusion comparison using the Archive scene, with stairs, shelves, and walls, plus 11 moving objects. Orbit the scene, switch among six AO modes, and inspect the live FPS counter or the AO mask.

Open `index.html` in a WebGL 2 browser. The scene and distance field are embedded. Three.js and its addons load from a pinned CDN URL.

## Moving objects

The static scene keeps its baked distance volume. Moving objects are not baked into it. The shader evaluates their distances at each AO tap and takes the minimum with the static volume, so moving or rotating an object needs no rebake.

- **Analytic shapes** use exact distance functions: box, sphere, capsule, cylinder, rounded box, and torus.
- **Rigid meshes of any shape** get a 64³ distance field baked once at load on the GPU, in the mesh's local space. Distance is exact point-to-triangle distance. The sign comes from the generalized winding number, so meshes that are not fully closed, such as the teapot, still bake correctly. All fields share one 3D atlas. Each field extends 1.2 m beyond its mesh, past the AO's reach, so nothing outside it can occlude. The demo bakes 11 fields with 18,476 triangles in 60–200 ms.
- **Culling** keeps the cost near the objects. Each draw instance first skips objects that cannot reach its bounds. Each pixel then skips any object whose bounding sphere the farthest tap, at 1.12 m along the normal, cannot reach. If the farthest tap cannot reach it, no nearer tap can. Convex shapes also skip themselves, because their own taps always point away from them. These tests only skip objects that cannot change the result. Renders matched pixel for pixel with them on and off.

Use **Movers in SDF AO** to compare with and without moving-object occlusion. **Baked SDF for all movers** switches the analytic shapes to their baked fields. Against the exact distance functions, about 8% of the pixels that moving objects affect change, by about 1/255 on average and at most 7/255.

Limits: up to 16 moving objects, rigid transforms only, and about 5 cm voxels in the demo's baked fields. Deforming meshes would need per-frame rebaking or a capsule approximation.

### Timings with all 11 objects moving

These are median GPU elapsed times per draw on an Apple M1 at 1280 × 720 over 25 rounds. The objects moved between rounds. They were measured in a visible browser tab with the render loop paused. This setup reads slower than the static benchmark below, which saw 0.16 ms for SDF wall correction. Here the SDF modes took 0.35–0.41 ms without moving objects in AO, with the objects still drawn. Compare rows within this table.

| Method | Median (ms) | p95 (ms) |
|---|---:|---:|
| AO off | 0.20 | 0.28 |
| SDF modes, no moving objects in AO | 0.35–0.41 | 0.42–0.49 |
| SDF modes, exact shapes | 1.33–1.36 | 1.54–1.61 |
| SDF modes, all baked fields | 1.33–1.39 | 1.51–1.75 |
| Three.js 0.170 SSAOPass (legacy, full resolution) | 8.68 | 10.57 |
| Three.js 0.170 GTAOPass (legacy, full resolution) | 20.72 | 24.38 |

Against these legacy Three.js 0.170 WebGL2 passes, SSAO and GTAO took about 6× and 15× as long as the SDF modes with all 11 objects moving. **These ratios overstate the gap to current Three.js.** The legacy passes run at full resolution, re-render the scene for their own normals and depth, and blur at full resolution. Three.js r187's WebGPU SSAO and GTAO (half resolution, MRT normals, TRAA) cost the same order of magnitude as the SDF AO; see [r187-results.md](benchmark/r187-results.md). SDF cost grows with the number of moving objects near each pixel. SSAO and GTAO cost the same regardless of what moves. Accuracy with moving objects has not been measured against ray-traced AO.

## Static benchmark

The [benchmark](benchmark/final-benchmark.md) compares four-tap SDF AO with the legacy Three.js 0.170 SSAO and GTAO passes on an Apple M1 in the static scene. A [comparison with Three.js r187's WebGPU SSAO and GTAO](benchmark/r187-results.md) finds similar GPU cost (about 0.2–0.3 ms per frame for every method) and lower AO error for the SDF: 0.040, against 0.051–0.053 for tuned GTAO + TRAA and 0.063–0.065 for tuned SSAO. It includes [GPU timings](benchmark/final-timings.json), [accuracy measurements](benchmark/final-accuracy.json), and a [side-by-side comparison](benchmark/final-comparison.png) against ray-traced AO. The live FPS counter measures browser frame cadence. The benchmark uses GPU elapsed queries.

The Archive volume was baked from analytic box distances. The runtime AO shader samples the baked volume.

## Baking a field for your own scene

[`tools/bake_sdf.cpp`](tools/bake_sdf.cpp) bakes the same 128³ clipped signed field from any closed triangle mesh (OBJ) or box list (JSON). It has no dependencies:

```
c++ -O3 -std=c++17 -pthread tools/bake_sdf.cpp -o bake_sdf
./bake_sdf --obj scene.obj --lo -1 -1 -1 --span 26 18 8 --res 128 --clip 1.5 --out scene.f32
```

Distances are exact (closest point on the nearest triangle), far space is skipped, sign comes from winding numbers, and all cores are used. On an Apple M1, the Archive takes 115 ms and a 488k-triangle version of it 307 ms. Outside the geometry the output matches the analytic field exactly, and AO accuracy is unchanged. See [precompute.md](benchmark/precompute.md).

DP-NNS was evaluated as an SDF baking accelerator in an earlier experiment. As a seed for this baker it cost more to build than it saved, so the tool doesn't use it.

[DP-NNS paper](https://arxiv.org/abs/2611.15320)
