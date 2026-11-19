#!/usr/bin/env python3
"""Independent 512-Ray Triangle Reference & Apple M1 GPU Benchmark for Curved Atrium.

Traces 512 cosine-weighted hemisphere rays per sampled pixel against the exact
Atrium triangle mesh via tools/trace_tris.cpp, then scores index.html (Atrium scene)
on real Apple M1 Chrome WebGL2 comparing:
  - Traced SDF + SAH BVH (frames 1 / 4 / 16 / 64, wall ms/f, BVH build ms & VRAM)
  - 4-Tap Original SDF (generic, no wallFace)
  - 4-Tap Wall-Correction SDF (wallFace = 0)
  - SSAO (mode 32)
  - GTAO (mode 36)

Also verifies 0 WebGL errors & 0 console errors across Archive -> Atrium -> Archive
switching and all toggles (#pbr, #hizcontact, #mask, #smoothfalloff), and saves
a visual PBR render to output/nextgen/atrium_pbr.png.
"""
import functools
import http.server
import json
import math
import os
import socketserver
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output/nextgen"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRACER_BIN = OUT_DIR / "bin/trace_tris"

W, H = 1280, 720
AO_RADIUS = 1.12
RAYS_PER_PX = 512


def ensure_tracer():
    if TRACER_BIN.exists() and TRACER_BIN.stat().st_mtime >= (ROOT / "tools/trace_tris.cpp").stat().st_mtime:
        return TRACER_BIN
    TRACER_BIN.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["c++", "-O3", "-std=c++17", "-pthread"]
    if os.uname().sysname == "Darwin":
        try:
            sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
            inc = Path(sdk) / "usr/include/c++/v1"
            if inc.exists():
                cmd += ["-isystem", str(inc)]
        except Exception:
            pass
    cmd += [str(ROOT / "tools/trace_tris.cpp"), "-o", str(TRACER_BIN)]
    subprocess.check_call(cmd)
    return TRACER_BIN


def start_server():
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass
    handler = functools.partial(QuietHandler, directory=str(ROOT))
    socketserver.TCPServer.allow_reuse_address = True
    srv = socketserver.TCPServer(("127.0.0.1", 0), handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, port


def trace_reference_512(tris_path: Path, pos_xyz: np.ndarray, nor_xyz: np.ndarray) -> np.ndarray:
    """Trace 512 stratified cosine-weighted hemisphere rays per pixel against exact triangles."""
    tracer = ensure_tracer()
    n_px = len(pos_xyz)
    rng = np.random.default_rng(2026)

    # Build tangent frame (T, B, N) for each pixel
    n = nor_xyz / np.maximum(np.linalg.norm(nor_xyz, axis=1, keepdims=True), 1e-8)
    helper = np.where(np.abs(n[:, 0:1]) < 0.9, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))
    t = np.cross(n, helper)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-8)
    b = np.cross(n, t)

    # Stratified 16 x 32 grid on hemisphere = 512 rays per pixel
    su, sv = 16, 32
    iu, iv = np.meshgrid(np.arange(su), np.arange(sv), indexing="ij")
    iu = iu.ravel()  # (512,)
    iv = iv.ravel()

    # Batch pixels to keep temp file size modest
    ao_ref = np.zeros(n_px, dtype=np.float32)
    batch_px = 1024
    with tempfile.TemporaryDirectory(prefix="atrium-ref-") as tmp:
        rays_file = Path(tmp) / "rays.f32"
        hits_file = Path(tmp) / "hits.u8"
        for start in range(0, n_px, batch_px):
            end = min(n_px, start + batch_px)
            m = end - start
            u = (iu[None, :] + rng.random((m, RAYS_PER_PX))) / su
            v = (iv[None, :] + rng.random((m, RAYS_PER_PX))) / sv
            r = np.sqrt(np.clip(u, 0.0, 1.0))
            phi = 2.0 * np.pi * v
            lx = r * np.cos(phi)
            ly = r * np.sin(phi)
            lz = np.sqrt(np.clip(1.0 - u, 0.0, 1.0))

            dirs = (
                lx[:, :, None] * t[start:end, None, :]
                + ly[:, :, None] * b[start:end, None, :]
                + lz[:, :, None] * n[start:end, None, :]
            )
            orig = pos_xyz[start:end, None, :] + 0.0003 * n[start:end, None, :]
            rays = np.empty((m, RAYS_PER_PX, 7), dtype=np.float32)
            rays[:, :, 0:3] = orig
            rays[:, :, 3:6] = dirs
            rays[:, :, 6] = AO_RADIUS
            rays.tofile(rays_file)
            subprocess.check_call(
                [str(tracer), str(tris_path), str(rays_file), str(hits_file)],
                stdout=subprocess.DEVNULL,
            )
            hits = np.fromfile(hits_file, dtype=np.uint8).reshape(m, RAYS_PER_PX)
            ao_ref[start:end] = 1.0 - hits.mean(axis=1)
    return ao_ref


def main():
    with tempfile.TemporaryDirectory(prefix="atrium-tris-") as tmp:
        tris_path = Path(tmp) / "atrium_tris.f32"
        subprocess.check_call(
            ["node", "tools/gen_atrium.mjs", "--tris-only", "--export-tris", str(tris_path)],
            cwd=str(ROOT),
        )

        srv, port = start_server()
        url = f"http://127.0.0.1:{port}/index.html"
        console_errors = []
        page_errors = []

        with sync_playwright() as p:
            browser = p.chromium.launch(
                channel="chrome",
                headless=True,
                args=[
                    "--use-gl=angle",
                    "--use-angle=metal",
                    "--enable-webgl",
                    "--ignore-gpu-blocklist",
                ],
            )
            page = browser.new_page(viewport={"width": 1360, "height": 960})
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
            page.on("pageerror", lambda err: page_errors.append(str(err)))
            page.goto(url, wait_until="networkidle")
            page.wait_for_function("() => window.aoDemo !== undefined", timeout=30000)

            # Stop RAF loop for deterministic frame stepping
            page.evaluate("() => { aoDemo.renderer.setAnimationLoop(null); }")

            # 1. Verify Archive -> Atrium -> Archive -> Atrium switching & all modes/toggles (0 GL errors)
            switch_check = page.evaluate("""async () => {
                const gl = aoDemo.renderer.getContext();
                const drain = () => {
                    const errs = [];
                    let e;
                    while ((e = gl.getError()) !== gl.NO_ERROR) errs.push(e);
                    return errs;
                };
                drain();
                const allErrs = [];
                const check = (tag) => {
                    const e = drain();
                    if (e.length) allErrs.push({ tag, errors: e });
                };

                await aoDemo.setScene('atrium');
                const atriumMeta = {
                    triangles: aoDemo.contactResources.triangles,
                    nodes: aoDemo.contactResources.nodes,
                    bytes: aoDemo.contactResources.bytes,
                    buildMs: aoDemo.contactResources.buildMs,
                };
                for (const mode of [0, 4, 14, 34, 64, 32, 36]) {
                    aoDemo.select(mode);
                    aoDemo.render(mode);
                    check('atrium-mode-' + mode);
                }
                for (const flag of ['pbr', 'hizcontact', 'mask', 'smoothfalloff']) {
                    const el = document.querySelector('#' + flag);
                    el.checked = true; el.onchange({ target: el });
                    aoDemo.select(64);
                    aoDemo.render(64);
                    check('atrium-flag-' + flag);
                    el.checked = false; el.onchange({ target: el });
                }
                await aoDemo.setScene('archive');
                aoDemo.select(64); aoDemo.render(64);
                check('back-to-archive');
                await aoDemo.setScene('atrium');
                aoDemo.select(64); aoDemo.render(64);
                check('back-to-atrium');
                return { allErrs, atriumMeta };
            }""")
            gl_errors = switch_check["allErrs"]
            atrium_meta = switch_check["atriumMeta"]
            print(f"Atrium BVH: {atrium_meta['triangles']} tris, {atrium_meta['nodes']} nodes, "
                  f"{atrium_meta['bytes']/1024:.1f} KiB VRAM, build {atrium_meta['buildMs']:.1f} ms")
            print(f"Switch/mode/toggle GL errors: {len(gl_errors)}")

            # 2. Capture visual PBR + Multi-Bounce render at frame 64
            page.evaluate("""() => {
                aoDemo.setMask(false);
                aoDemo.setPBR(true);
                aoDemo.setContactMode(0);
                aoDemo.setSmoothFalloff(false);
                aoDemo.select(64);
                aoDemo.traced.frame.value = 0;
                aoDemo.resetHistory();
                for (let i = 0; i < 64; i++) aoDemo.render(64);
                aoDemo.renderer.getContext().finish();
            }""")
            canvas = page.locator("canvas")
            pbr_path = OUT_DIR / "atrium_pbr.png"
            canvas.screenshot(path=str(pbr_path))
            print(f"Saved visual capture: {pbr_path}")

            # 3. Extract full-res G-buffer (world pos, normal) on 4-stride grid (180x320) for 512-ray reference
            gbuf = page.evaluate("""() => {
                aoDemo.setPBR(false);
                aoDemo.setMask(true);
                aoDemo.setAOScale(1);
                aoDemo.select(64);
                aoDemo.traced.frame.value = 0;
                aoDemo.resetHistory();
                aoDemo.render(64);
                const gl = aoDemo.renderer.getContext();
                const T = aoDemo.T;
                // Read full-res world position & normal by rendering gTarget at scale 1
                const W = 1280, H = 720;
                const posRT = new T.WebGLRenderTarget(W, H, { type: T.FloatType, format: T.RGBAFormat });
                const norRT = new T.WebGLRenderTarget(W, H, { type: T.FloatType, format: T.RGBAFormat });
                const posMat = new T.ShaderMaterial({
                    glslVersion: T.GLSL3,
                    vertexShader: `out vec3 wp; void main(){ vec4 w = modelMatrix * instanceMatrix * vec4(position, 1.); wp = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
                    fragmentShader: `precision highp float; in vec3 wp; out vec4 o; void main(){ o = vec4(wp, 1.); }`
                });
                const norMat = new T.ShaderMaterial({
                    glslVersion: T.GLSL3,
                    vertexShader: `out vec3 wn; void main(){ wn = normalize(mat3(modelMatrix) * mat3(instanceMatrix) * normal); gl_Position = projectionMatrix * viewMatrix * modelMatrix * instanceMatrix * vec4(position, 1.); }`,
                    fragmentShader: `precision highp float; in vec3 wn; out vec4 o; void main(){ o = vec4(normalize(wn), 1.); }`
                });
                const savedBg = aoDemo.scene.background;
                const savedMat = aoDemo.atriumMesh.material;
                aoDemo.scene.background = null;
                aoDemo.renderer.setClearColor(0, 0);

                aoDemo.atriumMesh.material = posMat;
                aoDemo.renderer.setRenderTarget(posRT);
                aoDemo.renderer.render(aoDemo.scene, aoDemo.camera);
                const posBuf = new Float32Array(W * H * 4);
                aoDemo.renderer.readRenderTargetPixels(posRT, 0, 0, W, H, posBuf);

                aoDemo.atriumMesh.material = norMat;
                aoDemo.renderer.setRenderTarget(norRT);
                aoDemo.renderer.render(aoDemo.scene, aoDemo.camera);
                const norBuf = new Float32Array(W * H * 4);
                aoDemo.renderer.readRenderTargetPixels(norRT, 0, 0, W, H, norBuf);

                aoDemo.scene.background = savedBg;
                aoDemo.atriumMesh.material = savedMat;
                aoDemo.renderer.setRenderTarget(null);
                posRT.dispose(); norRT.dispose(); posMat.dispose(); norMat.dispose();

                // Restore half-res AO scale for actual production scoring
                aoDemo.setAOScale(0.5);

                const indices = [];
                const pos = [];
                const nor = [];
                for (let y = 2; y < H; y += 4) {
                    for (let x = 2; x < W; x += 4) {
                        const pIdx = y * W + x;
                        const o = pIdx * 4;
                        if (posBuf[o + 3] > 0.5) {
                            indices.push(pIdx);
                            pos.push(posBuf[o], posBuf[o + 1], posBuf[o + 2]);
                            nor.push(norBuf[o], norBuf[o + 1], norBuf[o + 2]);
                        }
                    }
                }
                return { indices, pos, nor };
            }""")

            indices = np.array(gbuf["indices"], dtype=np.int32)
            pos_xyz = np.array(gbuf["pos"], dtype=np.float32).reshape(-1, 3)
            nor_xyz = np.array(gbuf["nor"], dtype=np.float32).reshape(-1, 3)
            print(f"Tracing 512-ray exact triangle reference on {len(indices):,} sampled Atrium pixels...")
            t0 = time.perf_counter()
            ao_ref = trace_reference_512(tris_path, pos_xyz, nor_xyz)
            print(f"512-ray reference traced in {time.perf_counter() - t0:.2f} s (mean AO = {ao_ref.mean():.4f})")

            # 4. Score all AO methods on Apple M1 Chrome WebGL2
            def read_mask_samples(setup_js: str):
                vals = page.evaluate(f"""(indices) => {{
                    {setup_js}
                    const gl = aoDemo.renderer.getContext();
                    const buf = new Uint8Array(1280 * 720 * 4);
                    gl.readPixels(0, 0, 1280, 720, gl.RGBA, gl.UNSIGNED_BYTE, buf);
                    const out = new Array(indices.length);
                    for (let i = 0; i < indices.length; i++) out[i] = buf[indices[i] * 4] / 255.0;
                    return out;
                }}""", indices.tolist())
                return np.array(vals, dtype=np.float32)

            results = {}

            # Traced SDF + SAH BVH at frames 1, 4, 16, 64
            traced_maes = {}
            traced_rmse = {}
            for f_target in [1, 4, 16, 64]:
                pred = read_mask_samples(f"""
                    aoDemo.setMask(true); aoDemo.setPBR(false); aoDemo.setContactMode(0);
                    aoDemo.select(64); aoDemo.traced.frame.value = 0; aoDemo.resetHistory();
                    for (let i = 0; i < {f_target}; i++) aoDemo.render(64);
                """)
                err = np.abs(pred - ao_ref)
                traced_maes[f_target] = float(err.mean())
                traced_rmse[f_target] = float(np.sqrt(np.mean((pred - ao_ref) ** 2)))
            results["traced_sdf_bvh"] = {"mae": traced_maes, "rmse": traced_rmse}

            # 4-Tap Original SDF (mode 4, generic)
            pred4 = read_mask_samples("aoDemo.setMask(true); aoDemo.setPBR(false); aoDemo.select(4); aoDemo.render(4);")
            results["sdf_4tap_original"] = {
                "mae": float(np.abs(pred4 - ao_ref).mean()),
                "rmse": float(np.sqrt(np.mean((pred4 - ao_ref) ** 2))),
            }

            # 4-Tap Wall-Correction SDF (mode 14, wallFace = 0 on Atrium)
            pred14 = read_mask_samples("aoDemo.setMask(true); aoDemo.setPBR(false); aoDemo.select(14); aoDemo.render(14);")
            results["sdf_4tap_wallcorr"] = {
                "mae": float(np.abs(pred14 - ao_ref).mean()),
                "rmse": float(np.sqrt(np.mean((pred14 - ao_ref) ** 2))),
            }

            # SSAO (mode 32)
            pred32 = read_mask_samples("aoDemo.setMask(true); aoDemo.setPBR(false); aoDemo.select(32); aoDemo.render(32);")
            results["ssao"] = {
                "mae": float(np.abs(pred32 - ao_ref).mean()),
                "rmse": float(np.sqrt(np.mean((pred32 - ao_ref) ** 2))),
            }

            # GTAO (mode 36)
            pred36 = read_mask_samples("aoDemo.setMask(true); aoDemo.setPBR(false); aoDemo.select(36); aoDemo.render(36);")
            results["gtao"] = {
                "mae": float(np.abs(pred36 - ao_ref).mean()),
                "rmse": float(np.sqrt(np.mean((pred36 - ao_ref) ** 2))),
            }

            # 5. Measure wall-clock ms/f (settled & early) for Traced SDF + 4-Tap + SSAO + GTAO
            timings = page.evaluate("""() => {
                const gl = aoDemo.renderer.getContext();
                const px = new Uint8Array(4);
                const sync = () => gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
                const measure = (fn, reps = 20) => {
                    for (let i = 0; i < 4; i++) fn();
                    sync();
                    const samples = [];
                    for (let i = 0; i < reps; i++) {
                        const t0 = performance.now();
                        fn();
                        sync();
                        samples.push(performance.now() - t0);
                    }
                    samples.sort((a, b) => a - b);
                    return samples[Math.floor(samples.length * 0.25)];
                };
                aoDemo.setMask(false); aoDemo.setPBR(false); aoDemo.setContactMode(0);

                // Settled Traced SDF (after 70 frames)
                aoDemo.select(64); aoDemo.traced.frame.value = 0; aoDemo.resetHistory();
                for (let i = 0; i < 70; i++) aoDemo.render(64);
                const tracedSettled = measure(() => aoDemo.render(64), 25);

                // Early Traced SDF (frame 0..3)
                const tracedEarly = measure(() => {
                    aoDemo.traced.frame.value = 0; aoDemo.resetHistory(); aoDemo.render(64);
                }, 15);

                aoDemo.select(4);
                const sdf4 = measure(() => aoDemo.render(4), 25);
                aoDemo.select(32);
                const ssaoMs = measure(() => aoDemo.render(32), 15);
                aoDemo.select(36);
                const gtaoMs = measure(() => aoDemo.render(36), 15);

                // 6. High-poly scaling stress test: 1.9k (Archive) -> 13.7k (Atrium) -> 54.8k (Subdivided Atrium)
                const scaling = [];
                scaling.push({
                    scene: 'Atrium (13.7k)',
                    triangles: aoDemo.contactResources.triangles,
                    gridSize: aoDemo.contactResources.gridSize,
                    nodes: aoDemo.contactResources.nodes,
                    vramKiB: +(aoDemo.contactResources.bytes / 1024).toFixed(1),
                    buildMs: +aoDemo.contactResources.buildMs.toFixed(1),
                    earlyMs: +tracedEarly.toFixed(2),
                    settledMs: +tracedSettled.toFixed(2),
                });
                const T = aoDemo.T;
                const srcGeo = aoDemo.atriumMesh.geometry;
                const pos = srcGeo.attributes.position.array;
                const idx = srcGeo.index.array;
                const subPos = new Float32Array((idx.length / 3) * 4 * 9);
                let o = 0;
                const pushTri = (ax,ay,az, bx,by,bz, cx,cy,cz) => {
                    subPos[o++]=ax; subPos[o++]=ay; subPos[o++]=az;
                    subPos[o++]=bx; subPos[o++]=by; subPos[o++]=bz;
                    subPos[o++]=cx; subPos[o++]=cy; subPos[o++]=cz;
                };
                for (let i = 0; i < idx.length; i += 3) {
                    const a = idx[i]*3, b = idx[i+1]*3, c = idx[i+2]*3;
                    const ax=pos[a], ay=pos[a+1], az=pos[a+2];
                    const bx=pos[b], by=pos[b+1], bz=pos[b+2];
                    const cx=pos[c], cy=pos[c+1], cz=pos[c+2];
                    const abx=.5*(ax+bx), aby=.5*(ay+by), abz=.5*(az+bz);
                    const bcx=.5*(bx+cx), bcy=.5*(by+cy), bcz=.5*(bz+cz);
                    const cax=.5*(cx+ax), cay=.5*(cy+ay), caz=.5*(cz+az);
                    pushTri(ax,ay,az, abx,aby,abz, cax,cay,caz);
                    pushTri(abx,aby,abz, bx,by,bz, bcx,bcy,bcz);
                    pushTri(cax,cay,caz, bcx,bcy,bcz, cx,cy,cz);
                    pushTri(abx,aby,abz, bcx,bcy,bcz, cax,cay,caz);
                }
                const hiGeo = new T.BufferGeometry();
                hiGeo.setAttribute('position', new T.BufferAttribute(subPos, 3));
                const hiMesh = new T.Mesh(hiGeo);
                hiMesh.updateMatrixWorld(true);
                const range = aoDemo.contactResources.range;
                const hiMeta = aoDemo.rebuildStaticContact([hiMesh], { gridSize: 'auto', range, bias: 0.0003 });
                aoDemo.select(64); aoDemo.traced.frame.value = 0; aoDemo.resetHistory();
                for (let i = 0; i < 70; i++) aoDemo.render(64);
                const hiSettled = measure(() => aoDemo.render(64), 20);
                const hiEarly = measure(() => { aoDemo.traced.frame.value = 0; aoDemo.resetHistory(); aoDemo.render(64); }, 15);
                scaling.push({
                    scene: 'Atrium-55k (subdiv)',
                    triangles: hiMeta.triangles,
                    gridSize: hiMeta.gridSize,
                    nodes: hiMeta.nodes,
                    vramKiB: +(hiMeta.bytes / 1024).toFixed(1),
                    buildMs: +hiMeta.buildMs.toFixed(1),
                    earlyMs: +hiEarly.toFixed(2),
                    settledMs: +hiSettled.toFixed(2),
                });
                hiGeo.dispose();
                aoDemo.rebuildStaticContact([aoDemo.atriumMesh], { gridSize: 'auto', range, bias: 0.0003 });

                return { tracedEarly, tracedSettled, sdf4, ssaoMs, gtaoMs, scaling };
            }""")

            browser.close()
        srv.shutdown()

    summary = {
        "atrium_meta": atrium_meta,
        "sampled_pixels": int(len(indices)),
        "rays_per_pixel": RAYS_PER_PX,
        "results": results,
        "timings_ms": timings,
        "gl_errors": gl_errors,
        "console_errors": console_errors,
        "page_errors": page_errors,
    }
    out_json = OUT_DIR / "atrium_benchmark.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== Curved Atrium Benchmark (Apple M1 Chrome WebGL2, 1280x720, 512-ray exact triangle ref) ===")
    print(f"| Method | MAE | RMSE | Wall ms/f |")
    print(f"|---|---:|---:|---:|")
    print(f"| Traced SDF + SAH BVH (frame 1 / 4 / 16 / 64) | "
          f"{traced_maes[1]:.5f} / {traced_maes[4]:.5f} / {traced_maes[16]:.5f} / **{traced_maes[64]:.5f}** | "
          f"{traced_rmse[64]:.5f} | {timings['tracedEarly']:.2f} early / **{timings['tracedSettled']:.2f}** settled |")
    print(f"| 4-Tap Original SDF (generic) | {results['sdf_4tap_original']['mae']:.5f} | {results['sdf_4tap_original']['rmse']:.5f} | {timings['sdf4']:.2f} |")
    print(f"| 4-Tap Wall-Correction (wallFace=0) | {results['sdf_4tap_wallcorr']['mae']:.5f} | {results['sdf_4tap_wallcorr']['rmse']:.5f} | {timings['sdf4']:.2f} |")
    print(f"| SSAO (Three.js 0.170) | {results['ssao']['mae']:.5f} | {results['ssao']['rmse']:.5f} | {timings['ssaoMs']:.2f} |")
    print(f"| GTAO (Three.js 0.170) | {results['gtao']['mae']:.5f} | {results['gtao']['rmse']:.5f} | {timings['gtaoMs']:.2f} |")
    print("\n=== High-Poly Scaling Stress Test ===")
    print("| Scene | Triangles | Grid | Nodes | VRAM (MiB) | Build (ms) | Early (ms/f) | Settled (ms/f) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in timings.get("scaling", []):
        print(f"| {s['scene']} | {s['triangles']:,} | {s['gridSize']:.2f} m | {s['nodes']:,} | {s['vramKiB']/1024:.2f} | {s['buildMs']:.1f} | {s['earlyMs']:.2f} | {s['settledMs']:.2f} |")
    print(f"\nGL errors: {len(gl_errors)} | Console errors: {len(console_errors)} | Page errors: {len(page_errors)}")
    if gl_errors or console_errors or page_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
