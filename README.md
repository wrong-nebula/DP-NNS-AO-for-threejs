# DP-NNS AO for Three.js

An interactive Three.js ambient occlusion comparison using a static Archive scene with stairs, shelves, and walls. Orbit the scene, switch among six AO modes, and inspect the live FPS counter or the AO mask.

Open `index.html` in a WebGL 2 browser. The scene and distance field are embedded. Three.js loads from a pinned CDN URL.

The [benchmark](benchmark/final-benchmark.md) compares four-tap SDF AO with the Three.js SSAO and GTAO passes on an Apple M1. It includes [GPU timings](benchmark/final-timings.json), [accuracy measurements](benchmark/final-accuracy.json), and a [side-by-side comparison](benchmark/final-comparison.png) against ray-traced AO. The live FPS counter measures browser frame cadence. The benchmark uses GPU elapsed queries.

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
