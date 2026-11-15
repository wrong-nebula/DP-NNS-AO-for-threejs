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

## Status: staged, not started

Decided: seed is the local commit `d81792b` on `speed-prototypes`, the model is Claude Opus 5.5, and the budget is $20 and 4 h.

Staged in `../thermite-staging/` (outside the repo):
- `dpnns-ao-seed.bundle`: seed commit `d81792ba4f2963c5ecaa38714ea3e5d700419553`.
- `bake-sdf-kit.bundle`: the kit as its own repo (`../thermite-kit-bake-sdf`), commit `4a5b0bc69ca6f485a8e9c3a07b81d3b82de1fa97`.
- `assign-payload.json`: built with `thermite assign-payload` with nothing issued. It carries both bundles as imports.

Not started because `local-1`, the only enrolled rig, is running another session's campaign (`procgen-boolean-quality-2`, assigned 12:51). A second judge on the same 6 CPUs would corrupt both campaigns' timings.

To start once the rig is free:
1. Copy both bundles into `~/thermite/.dev/inbox/repos/`.
2. Issue the payload: `thermite command --kind assign_campaign --payload @assign-payload.json --rig local-1 --control http://127.0.0.1:8800 --token-file <owner token>`.
3. If it doesn't start agents itself, give the model with `run start bake-sdf-speed --rig local-1 --provider <anthropic provider name in control Settings> --model claude-opus-5-5 …`.

The rig's `model-api-key` file is 17 bytes, which looks like a placeholder, so a provider must be set up in control's Settings first.

GPU work (runtime AO, the mover bake) can't run on this rig: it is a CPU target in a Linux container with no browser GPU. Those stay with `benchmark/mover-bench.js` on the Mac.
