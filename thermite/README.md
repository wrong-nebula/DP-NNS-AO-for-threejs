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

## Status: running on its own rig, `local-4`

- **Run `bake-sdf-speed-4`** was assigned 2026-11-15 13:54 JST, from contract v4 (kit `5126ae9`: one third of the CPU quota as bake threads). It is seeded from `d81792b`, with a budget of $20 and 4 h.
- **Rig `local-4`:** added for this campaign only, so it never displaces another campaign. It is defined in `~/thermite/compose.override.yaml` (profile `rig-four`, derived from the `-3` services), with the harness at 3 CPUs / 2.5 GB and the supervisor at 1 CPU. The images are the stock 0.2.4 harness and supervisor, each with only Claude Code 2.1.325 added. Workers run `opus-5.5` from the `default` provider (your Claude subscription) under Claude Code. With a 3-CPU quota, the bake gets 1 thread.
- **Baseline** (13:55, 3 blocks): archive `bake_s` 0.726 s ±16%, dense 1.219 s ±4.5%. MAE 0.04012 / 0.04211 and `visible_mismatch` 0 on both cells. Calibration (A/A) and the planner's epoch 1 started right after.
- **Earlier runs, kept for the record:** 1, no kit named in the contract; 2 and 3, baseline blocks CPU-throttled (8, then 4 threads on `local-1`'s 6-CPU quota, on a shared machine at load 11–15); 3 was also displaced on `local-1` by another session's run.
- **Results:** follow the run on the dashboard (http://127.0.0.1:8800/ui) or at `GET /v1/runs/bake-sdf-speed-4`.

GPU work (runtime AO, the mover bake) can't run on this rig: it is a CPU target in a Linux container with no browser GPU. Those stay with `benchmark/mover-bench.js` on the Mac.
