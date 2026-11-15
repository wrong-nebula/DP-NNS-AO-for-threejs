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

## Status: running

- **Run `bake-sdf-speed-2`** was assigned to `local-1` on 2026-11-15 at 13:10 JST, from contract v2 (sha256 `ced2fc07…`). It is seeded from `d81792b`, uses kit `4a5b0bc`, and is capped at $20 and 4 h.
- **Agents:** Claude Code 2.1.325 on a Claude subscription (`claude-code:opus-5.5`, set on the rig's Workers line). The rig's harness and supervisor images add only the `claude` binary; see the comments in `~/thermite/compose.override.yaml`.
- **Run `bake-sdf-speed-1`** failed its baseline jobs ("contract names no kit"): v1 of the contract had no `project.kit`. v2 names the seed and kit by the repos the assign payload imports (`dpnns-ao`, `bake-sdf-kit`).
- **Staging:** bundles and payloads are in `../thermite-staging/`, and the bundles also sit in `~/thermite/.dev/inbox/repos/`.
- **Results:** follow the run on the dashboard (http://127.0.0.1:8800/ui). Its champions land in the rig's results repository.

GPU work (runtime AO, the mover bake) can't run on this rig: it is a CPU target in a Linux container with no browser GPU. Those stay with `benchmark/mover-bench.js` on the Mac.
