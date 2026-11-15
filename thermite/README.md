# Thermite campaign: static bake speed

A hill climb on `tools/bake_sdf.cpp`: lower bake wall time with output that the AO cannot tell apart from the original baker's.

- [bake-sdf-speed.contract.json](bake-sdf-speed.contract.json) validates with `thermite contract validate`. Its only warning is the missing kit commit, which is given at assign time.
- [bake-sdf-kit/](bake-sdf-kit/) validates against Thermite's `KitSchema`. Its contents:

| Kit part | What it does |
|---|---|
| build | `c++ -O3 -std=c++17 -pthread tools/bake_sdf.cpp -o bake_sdf` |
| cells | `archive` (1,908 triangles), `dense` (488k), held-out `dense8` (122k) |
| `bake_s` | wall time of one bake, by the judge's clock. The primary metric, min, 3% minimum effect. |
| `visible_mismatch` | voxels that differ from the **original** baker's output where either value is below 1.33 m, the band trilinear AO samples can read. Hard constraint: 0. |
| `mae14`, `mae4` | static AO MAE against the 512-ray reference, from `eval_ao.py`. Hard constraints: ≤ 0.0402 and ≤ 0.0422. |

The reference fields, `eval_ao.py` and the ray-traced reference live in `refs/`. Only the judge's check reads them; runs copy just `run.sh` and the scene. `forbidden_patterns` refuse candidates that mention them.

## Dry run in the rig's harness image (6 CPUs, as `compose.override.yaml` caps it)

| Arm | archive | dense | dense8 | check |
|---|---:|---:|---:|---|
| original upstream baker | 234 ms | 511 ms | 304 ms | pass, mismatch 0, MAE 0.04012 / 0.04211 |
| `speed-prototypes` branch | 182 ms | 562 ms | 279 ms | pass, mismatch 0, same MAE |

Single runs on a shared host; the campaign uses 3 runs per cell per block, and 5 confirmation blocks. Negative tests: one voxel changed by 1 mm inside the band fails, and far values raised to the clip pass.

## Status: run 5, clang judge, on `local-4`

- **Run `bake-sdf-speed-5`** was assigned 2026-11-15 18:15 JST, from contract v5. Kit `b10c26d` builds with **clang++ 16 + libc++** (Debian), the closest match to the target Mac's Apple clang 16 + libc++. It is seeded from champion #3 (`07382ae`).
- **Why the switch:** run 4's judge built with g++ 12. It promoted a 4-wide BVH (its champion #4) that is 7–24% slower than champion #3 under Apple clang at 1 and 8 threads, with identical output. The clang judge, on cores 6–7, ranks the two the way the Mac does: archive +28.5% and dense +9% slower for #4.
- **Rig `local-4`:** harness image `local/thermite-harness:0.2.4-claude-clang`, pinned to cores 6–7 while `local-2` is stopped, with no CPU quota. Workers run `opus-5.5` (Claude subscription) under Claude Code.
- **Run 4 (g++ judge), paused:** champions #2 binned-SAH (`e6483a1` here) and #3 cache-line nodes (`07382ae` here) were verified on the Mac and committed; #4 was not adopted. Trials: 10, with 3 promoted, 3 not confirmed and 4 inconclusive.

GPU work (runtime AO, the mover bake) can't run on this rig: it is a CPU target in a Linux container with no browser GPU. Those stay with `benchmark/mover-bench.js` on the Mac.
