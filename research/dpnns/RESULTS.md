# DP-NNS vs kd-tree vs point BVH for baking AO fields from point clouds

The question: if the scene arrives as an oriented point cloud (a scan, photogrammetry, splats) instead of
triangles, is DP-NNS ([arXiv 2611.15320](https://arxiv.org/abs/2611.15320)) the best way to bake the
128³ SDF this viewer's AO reads? And what does baking from points cost compared with the exact triangle
baker (`tools/bake_sdf.cpp`)?

## Setup

- **Scene:** the Archive (159 boxes). Faces are sampled area-uniformly, with normals; samples on surfaces
  covered by another box are dropped, as a scan wouldn't see them. Targets of 100k / 300k / 1M / 3M samples
  leave 85k / 256k / 851k / 2.55M points.
- **Queries:** the 2,097,152 voxel centres of the 128³ bake grid, in 3D Morton order (the sort is counted in
  the query time). Signed distance = distance to the nearest point, signed by its normal, clipped to ±1.5
  like `bake_sdf`.
- **DP-NNS** (`pointnn_bench.cpp`), written from the paper (it publishes no code):
  - CGAL 6.2.1 exact-predicate Delaunay; oneTBB 2023.2;
  - BRIO levels at ratio 1/8, Hilbert-sorted;
  - successor DAG recorded at insertion;
  - level-consistent parallel build with CGAL's priority-blocking vertex locks, deferred retries and a
    barrier per level;
  - grid start (the highest-rank site shared by the 8 corner paths), G = 0 / 32 / 64.
- **Baselines:**
  - nanoflann 2.0.1 kd-tree (leaf 10, parallel build);
  - a median-split BVH over points (leaf 4, parallel build, nearest-child-first traversal).
- **Exactness:** all three methods return the exact nearest point for **all 2.1M queries** (checked against
  brute force), and their fields are byte-identical. So speed is the only difference between them.
- **Machine:** Apple M-series, 8 cores, shared with other sessions. The load average ranged 18–92 during
  the matrix. Each figure is the median of 3 interleaved rounds. The 8-thread figures are noisy and
  understate parallel scaling; the 3M row is a single run at load ~14.

Build and reproduce (CGAL, oneTBB and nanoflann from Homebrew; macOS may also need `-isysroot $(xcrun --show-sdk-path)`):

```bash
c++ -O3 -std=c++17 -DNDEBUG -DCGAL_NDEBUG -I/opt/homebrew/include research/dpnns/pointnn_bench.cpp \
    -L/opt/homebrew/lib -ltbb -ltbbmalloc -lgmp -lmpfr -o pointnn_bench
```

Then:

```bash
bash research/dpnns/run_bench.sh ./pointnn_bench ./bake_sdf 3 > runs.jsonl
```

## Speed (ms, 2.1M queries)

| Points | Threads | kd-tree build + query | BVH build + query | DP-NNS G=64 build + query | DP-NNS query µs | kd µs | BVH µs |
|---|---|---|---|---|---|---|---|
| 85k  | 1 | 11 + 1660 = **1672** | 20 + 1661 = 1681 | 548 + 352 = **900** | 0.17 | 0.79 | 0.79 |
| 85k  | 8 | 4 + 390 = 394 | 12 + 355 = **367** | 399 + 142 = 541 | 0.07 | 0.19 | 0.17 |
| 256k | 1 | 47 + 2429 = 2476 | 65 + 1789 = **1854** | 1535 + 443 = 1978 | 0.21 | 1.16 | 0.85 |
| 256k | 8 | 14 + 595 = 608 | 23 + 387 = **410** | 1121 + 150 = 1271 | 0.07 | 0.28 | 0.18 |
| 851k | 1 | 150 + 4114 = 4264 | 290 + 2099 = **2389** | 5207 + 814 = 6022 | 0.39 | 1.96 | 1.00 |
| 851k | 8 | 50 + 1137 = 1187 | 113 + 718 = **831** | 3349 + 229 = 3578 | 0.11 | 0.54 | 0.34 |
| 2.55M | 8 | 268 + 2064 = 2332 | 481 + 835 = **1316** | 16341 + 454 = 16795 | 0.22 | 0.98 | 0.40 |

Grid start, per query (851k points, 1 thread):

| Start | Distance checks | Query time |
|---|---|---|
| p₁ (no grid) | 443 | 1082 ms |
| G = 32 | 312 | 894 ms |
| G = 64 | 246 | 814 ms |

The grid itself costs 8–300 ms to build (G = 64 on one thread is the slow end).

**The exact triangle baker on the same grid:**

| Threads | With Lipschitz culling (shipped) | Every voxel |
|---|---|---|
| 1 | 68 ms | 133 ms |
| 8 | **30 ms** | 139 ms |

## Accuracy (identical for all three point methods)

| Points | Field mean / p95 error (reach band) | Wrong-sign voxels | AO MAE, 4-tap | AO MAE, wall correction |
|---|---|---|---|---|
| exact triangles | 0 / 0 mm | 0 | **0.0421** | **0.0401** |
| 85k   | 31.1 / 87.4 mm | 0.95% | 0.0558 (+33%) | 0.0539 (+34%) |
| 256k  | 15.6 / 42.5 mm | 0.39% | 0.0468 (+11%) | 0.0447 (+11%) |
| 851k  | 8.3 / 19.0 mm  | 0.12% | 0.0436 (+4%)  | 0.0416 (+4%)  |
| 2.55M | 5.9 / 9.0 mm   | 0.06% | 0.0423 (+0.5%) | 0.0404 (+0.6%) |

- **Reach band:** voxels with |d| < 1.2 m, the AO's largest tap.
- **AO MAE:** `benchmark/eval_ao.py` against the 512-ray reference. Its "wall correction" column is the
  older formula that script still evaluates, not the current shader's.
- **Wrong signs** come from contacts, where the nearest sample belongs to the neighbouring surface. That
  can't happen with a mesh and its winding number.

## Findings

1. **DP-NNS is the fastest query, as the paper says.** Its queries are 2.7–5× faster than nanoflann and
   1.8–4.6× faster than a point BVH. It is exact and stays exact under the parallel build. The grid start
   cuts distance checks by 30–45% (the paper: 325 → 143 on Lucy).
2. **It loses every complete bake here, because its build dominates.** The build is 18–50× slower than a
   BVH's, and one 128³ bake is only 2.1M queries.
   - DP-NNS only finishes first single-threaded at 85k points: 900 vs 1672 ms.
   - At 851k points on 8 threads, the BVH needs 831 ms and DP-NNS 3578 ms.
   - DP-NNS's build pays off after about 8–14M queries on one index, a 200³–240³ bake or several bakes of
     one point cloud. The paper's ICP use case is that regime: many queries against one fixed cloud.
3. **Queries per point grew with point count** (247 → 673 checks from 85k to 2.55M), where the paper
   argues O(log n). The likely cause is that box faces are exactly coplanar samples, so the Delaunay
   triangulation is heavily degenerate. Curved scans, the paper's datasets, should behave better. The
   per-query lead held anyway.
4. **Baking from points instead of triangles costs accuracy and time.**
   - It takes about 2.5M samples to get within 1% of the exact field's AO error.
   - Even then, the fastest point bake (BVH, 1.3 s) is about 40× slower than the exact triangle baker
     (30 ms), which also needs no sampling and has no sign errors.

**Recommendation:**
- When the input is a mesh, keep the triangle baker.
- For point-cloud input, use a BVH over points for a single bake. This is now `bake_sdf --points`, which adds the
  baker's Lipschitz culling and was 2–3× faster than this benchmark's BVH under the same load.
- Use DP-NNS only when one cloud serves many bakes or very dense grids (≳ 200³), or for streaming queries
  against a fixed cloud.
