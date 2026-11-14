# DP-NNS AO for Three.js

An interactive Three.js ambient occlusion comparison using a static Archive scene with stairs, shelves, and walls. Orbit the scene, switch among six AO modes, and inspect the live FPS counter or the AO mask.

Open `index.html` in a WebGL 2 browser. The scene and distance field are embedded. Three.js loads from a pinned CDN URL.

The [benchmark](benchmark/final-benchmark.md) compares four-tap SDF AO with the Three.js SSAO and GTAO passes on an Apple M1. It includes [GPU timings](benchmark/final-timings.json), [accuracy measurements](benchmark/final-accuracy.json), and a [side-by-side comparison](benchmark/final-comparison.png) against ray-traced AO. The live FPS counter measures browser frame cadence. The benchmark uses GPU elapsed queries.

The Archive volume was baked from analytic box distances. DP-NNS was evaluated as an SDF baking accelerator in the earlier experiment. The runtime AO shader samples the baked volume.

[DP-NNS paper](https://arxiv.org/abs/2611.15320)
