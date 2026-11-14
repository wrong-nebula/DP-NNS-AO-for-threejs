# AO field precompute

[`tools/bake_sdf.cpp`](../tools/bake_sdf.cpp) bakes the clipped signed distance volume that the SDF AO shader samples, from any closed triangle mesh. It is a single C++17 file with no dependencies.

```
c++ -O3 -std=c++17 -pthread tools/bake_sdf.cpp -o bake_sdf
./bake_sdf --boxes benchmark/archive.json --lo -1 -1 -1 --span 26 18 8 --res 128 --clip 1.5 --out archive.f32
./bake_sdf --obj my-scene.obj            --lo ... --span ...           --res 128 --clip 1.5 --out my-scene.f32
```

The output has the same layout as `archive.f32`: float32, x fastest, then y, then z. Voxel *i* sits at `lo + (i + 0.5) * span / res`. Values are negative inside and clipped at `+clip` outside. The shipped `archive.f32` and the viewer are unchanged.

## How it works

- **Exact distance.** Each voxel gets the distance to the closest point on the nearest triangle, not to surface samples. A BVH answers the query, starting from the previous voxel's nearest triangle as the bound, so most of the tree is skipped.
- **Far space skipped exactly.** The volume is processed in 8³ blocks. If a block's centre is farther than `clip` plus the block's half-diagonal from every triangle, every voxel in the block is beyond `clip`. Distance changes by at most the distance moved, so the whole block is filled without per-voxel queries.
- **Sign from the winding number.** A ray along z runs through each voxel column, counting +1 for each surface it enters and −1 for each it leaves. A voxel is inside where the count is nonzero. This handles overlapping objects, such as posts through shelves, without knowing which triangles belong to which object. Meshes must be closed and consistently wound.
- **All cores.** Blocks are shared across threads.

## Results (Apple M1, 8 threads unless noted)

**Exactness on the Archive scene (159 boxes).**
- Outside the geometry, the baked field matches the analytic box distances exactly (maximum difference 0.0), and every voxel has the same sign.
- The same holds when each box face is split into a 16 × 16 grid (488,448 triangles), and when the scene is read from OBJ instead of box JSON. The OBJ and box bakes are bit-identical.
- Inside overlapping boxes, both fields approximate the true distance, so values differ (mean 0.8 mm, maximum 18 cm).

**AO quality is unchanged.** Evaluating the viewer's shader at the 57,216 benchmark pixels against the 512-ray reference:

| AO mode | MAE, shipped field | MAE, baked field | pixels changed by > 0.01 |
|---|---|---|---|
| Original 4-tap SDF | 0.0421 | 0.0421 | 11 |
| SDF wall correction | 0.0401 | 0.0401 | 13 |

**Speed**

| scene | this baker | 1 thread | previous method |
|---|---|---|---|
| Archive, 1,908 triangles | **115 ms** | 500 ms | analytic numpy script: 120–190 ms, boxes only |
| Archive as a dense mesh, 488,448 triangles | **307 ms** | 884 ms | not possible with the analytic script |
| earlier 192-box test, 250k surface samples | — | — | DP-NNS sample bake: 0.61 s plus a 3.5 s index build, with 3.7 cm p95 error |

Skipping far blocks matters: without it, the distance pass takes 241 ms (vs 109 ms) on the Archive and 365 ms (vs 146 ms) on the dense mesh.

**Search methods (Archive, 8 threads, total ms)**

| triangles | uniform grid | BVH | BVH + previous voxel's triangle | BVH + nanoflann sample seed | BVH + DP-NNS sample seed |
|---|---|---|---|---|---|
| 1,908 | 122 | 143 | **115** | 393 | 571 |
| 488,448 | 1,766 | 343 | **307** | 680 | 1,610 |

Seeding from the nearest surface sample (nanoflann or DP-NNS, tested in a separate research build) trims the triangles tested per query from 8.6 to 7.2. But building the sample index costs more than it saves. The previous voxel's triangle is a free bound that is almost as tight, so this baker doesn't use a point index.

## Limits

- **Closed, consistently wound meshes only.** An open mesh, such as a raw scan with holes, gets correct distances but an unreliable sign.
- **Inside overlaps, the distance is to the nearest triangle,** which can be a face buried inside another object. That's the same kind of approximation as the analytic union. AO taps that land inside geometry are fully occluded either way.
- **One test machine and one scene,** in a dense and a sparse version. Timings exclude writing the file.
