#!/usr/bin/env python3
"""Independent 512-Ray 1-Bounce Diffuse GI + AO Benchmark & Visual Audit for Gallery.

Traces 512 stratified cosine-weighted 1-bounce GI + AO rays per sampled pixel
against the exact Gallery triangles via tools/trace_gi.cpp, then scores index.html
on real Apple M1 Chrome WebGL2:
  - Gallery AO-only vs Gallery 1-Bounce GI (RGB MAE, PSNR, bounce-only MAE, AO MAE)
  - Wall-clock ms/f (early & settled), radiance-grid build ms, relight ms & VRAM
  - 0 WebGL errors, 0 console errors, 0 page errors across all 3 scenes, 7 AO modes & all toggles
  - Saves 3 PNGs under output/nextgen/:
      gallery_ao_only.png
      gallery_1bounce_gi.png
      gallery_gi_only.png
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
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output/nextgen"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRACER_BIN = OUT_DIR / "bin/trace_gi"

W, H = 1280, 720


def ensure_gi_tracer():
    src = ROOT / "tools/trace_gi.cpp"
    if TRACER_BIN.exists() and TRACER_BIN.stat().st_mtime >= src.stat().st_mtime:
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
    cmd += [str(src), "-o", str(TRACER_BIN)]
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


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, np.power(np.maximum((c + 0.055) / 1.055, 0.0), 2.4))


def psnr(pred: np.ndarray, ref: np.ndarray) -> float:
    mse = float(np.mean((pred - ref) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


def main():
    tracer = ensure_gi_tracer()
    with tempfile.TemporaryDirectory(prefix="gallery-gi-") as tmp:
        gi_tris_path = Path(tmp) / "gallery_gi_tris.f32"
        subprocess.check_call(
            ["node", "tools/gen_gallery.mjs", "--tris-only", "--export-gi-tris", str(gi_tris_path)],
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
            page = browser.new_page(viewport={"width": 1360, "height": 980})
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
            page.on("pageerror", lambda err: page_errors.append(str(err)))
            page.goto(url, wait_until="networkidle")
            page.wait_for_function("() => window.aoDemo !== undefined", timeout=30000)
            page.evaluate("() => { aoDemo.renderer.setAnimationLoop(null); }")

            # 1. Exhaustive WebGL error check across all 3 scenes, 7 AO modes & all toggles
            check_res = page.evaluate("""async () => {
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

                for (const sc of ['archive', 'atrium', 'gallery']) {
                    await aoDemo.setScene(sc);
                    for (const mode of [0, 4, 14, 34, 64, 32, 36]) {
                        aoDemo.select(mode);
                        aoDemo.render(mode);
                        check(sc + '-mode-' + mode);
                    }
                    aoDemo.select(64);
                    for (const flag of ['pbr', 'hizcontact', 'mask', 'smoothfalloff', 'gi', 'gimask']) {
                        const el = document.querySelector('#' + flag);
                        const prev = el.checked;
                        el.checked = true; el.onchange({ target: el });
                        aoDemo.render(64);
                        check(sc + '-flag-on-' + flag);
                        el.checked = false; el.onchange({ target: el });
                        aoDemo.render(64);
                        check(sc + '-flag-off-' + flag);
                        el.checked = prev; el.onchange({ target: el });
                    }
                }
                // Test dynamic sun angle relight on Gallery
                await aoDemo.setScene('gallery');
                const relightTimes = [];
                for (const deg of [-45, -15, 0, 22, 45]) {
                    aoDemo.setSunAngle(deg);
                    relightTimes.push(aoDemo.radianceGrid.relightMs);
                    aoDemo.render(64);
                    check('gallery-sun-' + deg);
                }
                aoDemo.setSunAngle(22);

                const rg = aoDemo.radianceGrid;
                const bvh = aoDemo.contactResources;
                return {
                    allErrs,
                    relightTimes,
                    sunDir: aoDemo.sunDir,
                    radianceGrid: {
                        dims: rg.dims,
                        bytes: rg.bytes,
                        occupiedVoxels: rg.occupiedVoxels,
                        buildMs: rg.buildMs,
                        relightMs: rg.relightMs,
                    },
                    bvh: {
                        triangles: bvh.triangles,
                        nodes: bvh.nodes,
                        bytes: bvh.bytes,
                        buildMs: bvh.buildMs,
                    },
                };
            }""")
            gl_errors = check_res["allErrs"]
            rg_meta = check_res["radianceGrid"]
            bvh_meta = check_res["bvh"]
            sun_dir = check_res["sunDir"]
            print(f"Gallery BVH: {bvh_meta['triangles']} tris, {bvh_meta['nodes']} nodes, "
                  f"{bvh_meta['bytes']/1024:.1f} KiB VRAM, build {bvh_meta['buildMs']:.1f} ms")
            print(f"Radiance Grid: {rg_meta['dims']} ({rg_meta['bytes']/1024:.0f} KiB VRAM, "
                  f"{rg_meta['occupiedVoxels']} occupied voxels), build {rg_meta['buildMs']:.1f} ms, "
                  f"relight median {np.median(check_res['relightTimes']):.2f} ms")
            print(f"All-scenes/modes/toggles GL errors: {len(gl_errors)}, console errors: {len(console_errors)}, page errors: {len(page_errors)}")

            # 2. Capture 3 visual PNGs at frame 64 on Gallery (sunAngle = 22 deg)
            canvas = page.locator("canvas")

            # (a) Gallery AO-only (GI off)
            page.evaluate("""() => {
                aoDemo.setSunAngle(22);
                aoDemo.setMask(false);
                aoDemo.setGIMask(false);
                aoDemo.setPBR(false);
                aoDemo.setGI(false);
                aoDemo.select(64);
                aoDemo.traced.frame.value = 0;
                aoDemo.resetHistory();
                for (let i = 0; i < 64; i++) aoDemo.render(64);
                aoDemo.renderer.getContext().finish();
            }""")
            ao_only_png = OUT_DIR / "gallery_ao_only.png"
            canvas.screenshot(path=str(ao_only_png))
            print(f"Saved {ao_only_png}")

            # (b) Gallery 1-Bounce GI
            page.evaluate("""() => {
                aoDemo.setSunAngle(22);
                aoDemo.setMask(false);
                aoDemo.setGIMask(false);
                aoDemo.setGI(true);
                aoDemo.select(64);
                aoDemo.traced.frame.value = 0;
                aoDemo.resetHistory();
                for (let i = 0; i < 64; i++) aoDemo.render(64);
                aoDemo.renderer.getContext().finish();
            }""")
            gi_png = OUT_DIR / "gallery_1bounce_gi.png"
            canvas.screenshot(path=str(gi_png))
            print(f"Saved {gi_png}")

            # (c) Gallery pure GI bounce irradiance buffer
            page.evaluate("""() => {
                aoDemo.setGIMask(true);
                aoDemo.render(64);
                aoDemo.renderer.getContext().finish();
            }""")
            gi_only_png = OUT_DIR / "gallery_gi_only.png"
            canvas.screenshot(path=str(gi_only_png))
            print(f"Saved {gi_only_png}")
            page.evaluate("() => { aoDemo.setGIMask(false); }")

            # 3. Extract full-res G-buffer (pos, nor, albedo) on 4-stride grid for 512-ray GI reference
            gbuf = page.evaluate("""() => {
                const T = aoDemo.T;
                const W = 1280, H = 720;
                const posRT = new T.WebGLRenderTarget(W, H, { type: T.FloatType, format: T.RGBAFormat });
                const norRT = new T.WebGLRenderTarget(W, H, { type: T.FloatType, format: T.RGBAFormat });
                const albRT = new T.WebGLRenderTarget(W, H, { type: T.FloatType, format: T.RGBAFormat });
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
                const albMat = new T.ShaderMaterial({
                    glslVersion: T.GLSL3,
                    vertexShader: `in vec3 surfAlbedo; out vec3 va; void main(){ va = surfAlbedo; gl_Position = projectionMatrix * viewMatrix * modelMatrix * instanceMatrix * vec4(position, 1.); }`,
                    fragmentShader: `precision highp float; in vec3 va; out vec4 o; void main(){ o = vec4(va, 1.); }`
                });
                const savedBg = aoDemo.scene.background;
                const savedMat = aoDemo.galleryMesh.material;
                aoDemo.scene.background = null;
                aoDemo.renderer.setClearColor(0, 0);

                aoDemo.galleryMesh.material = posMat;
                aoDemo.renderer.setRenderTarget(posRT);
                aoDemo.renderer.render(aoDemo.scene, aoDemo.camera);
                const posBuf = new Float32Array(W * H * 4);
                aoDemo.renderer.readRenderTargetPixels(posRT, 0, 0, W, H, posBuf);

                aoDemo.galleryMesh.material = norMat;
                aoDemo.renderer.setRenderTarget(norRT);
                aoDemo.renderer.render(aoDemo.scene, aoDemo.camera);
                const norBuf = new Float32Array(W * H * 4);
                aoDemo.renderer.readRenderTargetPixels(norRT, 0, 0, W, H, norBuf);

                aoDemo.galleryMesh.material = albMat;
                aoDemo.renderer.setRenderTarget(albRT);
                aoDemo.renderer.render(aoDemo.scene, aoDemo.camera);
                const albBuf = new Float32Array(W * H * 4);
                aoDemo.renderer.readRenderTargetPixels(albRT, 0, 0, W, H, albBuf);

                aoDemo.scene.background = savedBg;
                aoDemo.galleryMesh.material = savedMat;
                aoDemo.renderer.setRenderTarget(null);
                posRT.dispose(); norRT.dispose(); albRT.dispose();
                posMat.dispose(); norMat.dispose(); albMat.dispose();

                const indices = [];
                const samples = [];
                for (let y = 2; y < H; y += 4) {
                    for (let x = 2; x < W; x += 4) {
                        const pIdx = y * W + x;
                        const o = pIdx * 4;
                        if (posBuf[o + 3] > 0.5) {
                            indices.push(pIdx);
                            samples.push(
                                posBuf[o], posBuf[o + 1], posBuf[o + 2],
                                norBuf[o], norBuf[o + 1], norBuf[o + 2],
                                albBuf[o], albBuf[o + 1], albBuf[o + 2]
                            );
                        }
                    }
                }
                return { indices, samples };
            }""")

            indices = np.array(gbuf["indices"], dtype=np.int32)
            samples = np.array(gbuf["samples"], dtype=np.float32)
            samples_path = Path(tmp) / "samples.f32"
            ref_path = Path(tmp) / "ref.f32"
            samples.tofile(samples_path)
            print(f"Tracing 512-ray 1-bounce GI + AO reference on {len(indices):,} sampled Gallery pixels...")
            t0 = time.perf_counter()
            subprocess.check_call([
                str(tracer), str(gi_tris_path), str(samples_path), str(ref_path),
                str(sun_dir[0]), str(sun_dir[1]), str(sun_dir[2]),
            ])
            print(f"512-ray GI reference traced in {time.perf_counter() - t0:.2f} s")

            ref = np.fromfile(ref_path, dtype=np.float32).reshape(-1, 10)
            ref_ao = ref[:, 0]
            ref_gi_irr = ref[:, 1:4]
            ref_rgb_ao_only = ref[:, 4:7]
            ref_rgb_gi = ref[:, 7:10]

            def read_rgb_samples(setup_js: str) -> np.ndarray:
                vals = page.evaluate(f"""(indices) => {{
                    {setup_js}
                    const gl = aoDemo.renderer.getContext();
                    const buf = new Uint8Array(1280 * 720 * 4);
                    gl.readPixels(0, 0, 1280, 720, gl.RGBA, gl.UNSIGNED_BYTE, buf);
                    const out = new Array(indices.length * 3);
                    for (let i = 0; i < indices.length; i++) {{
                        const o = indices[i] * 4;
                        out[i * 3]     = buf[o]     / 255.0;
                        out[i * 3 + 1] = buf[o + 1] / 255.0;
                        out[i * 3 + 2] = buf[o + 2] / 255.0;
                    }}
                    return out;
                }}""", indices.tolist())
                return np.array(vals, dtype=np.float32).reshape(-1, 3)

            # Score AO mask @64
            pred_ao = read_rgb_samples("""
                aoDemo.setGI(false); aoDemo.setGIMask(false); aoDemo.setMask(true);
                aoDemo.select(64); aoDemo.traced.frame.value = 0; aoDemo.resetHistory();
                for (let i = 0; i < 64; i++) aoDemo.render(64);
            """)[:, 0]
            ao_mae = float(np.abs(pred_ao - ref_ao).mean())

            # Score Gallery AO-only RGB @64 against 512-ray 1-bounce GI reference
            pred_rgb_ao = read_rgb_samples("""
                aoDemo.setMask(false); aoDemo.setGIMask(false); aoDemo.setGI(false);
                aoDemo.select(64); aoDemo.traced.frame.value = 0; aoDemo.resetHistory();
                for (let i = 0; i < 64; i++) aoDemo.render(64);
            """)

            # Score Gallery 1-Bounce GI RGB @64 against 512-ray 1-bounce GI reference
            pred_rgb_gi = read_rgb_samples("""
                aoDemo.setMask(false); aoDemo.setGIMask(false); aoDemo.setGI(true);
                aoDemo.select(64); aoDemo.traced.frame.value = 0; aoDemo.resetHistory();
                for (let i = 0; i < 64; i++) aoDemo.render(64);
            """)

            # Score pure GI bounce irradiance buffer @64
            pred_gi_srgb = read_rgb_samples("""
                aoDemo.setMask(false); aoDemo.setGIMask(true); aoDemo.setGI(true);
                aoDemo.select(64); aoDemo.traced.frame.value = 0; aoDemo.resetHistory();
                for (let i = 0; i < 64; i++) aoDemo.render(64);
            """)
            pred_gi_lin = srgb_to_linear(pred_gi_srgb)
            bounce_mae = float(np.abs(pred_gi_lin - np.clip(ref_gi_irr, 0.0, 1.0)).mean())

            ao_only_mae_vs_gi_ref = float(np.abs(pred_rgb_ao - ref_rgb_gi).mean())
            ao_only_psnr_vs_gi_ref = psnr(pred_rgb_ao, ref_rgb_gi)
            gi_mae_vs_gi_ref = float(np.abs(pred_rgb_gi - ref_rgb_gi).mean())
            gi_psnr_vs_gi_ref = psnr(pred_rgb_gi, ref_rgb_gi)

            # 4. Measure wall-clock ms/f (early & settled) for Gallery AO-only vs Gallery 1-Bounce GI
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
                aoDemo.setMask(false); aoDemo.setGIMask(false); aoDemo.setPBR(false); aoDemo.select(64);

                // Gallery AO-only
                aoDemo.setGI(false);
                aoDemo.traced.frame.value = 0; aoDemo.resetHistory();
                for (let i = 0; i < 70; i++) aoDemo.render(64);
                const aoSettled = measure(() => aoDemo.render(64), 25);
                const aoEarly = measure(() => {
                    aoDemo.traced.frame.value = 0; aoDemo.resetHistory(); aoDemo.render(64);
                }, 15);

                // Gallery 1-Bounce GI
                aoDemo.setGI(true);
                aoDemo.traced.frame.value = 0; aoDemo.resetHistory();
                for (let i = 0; i < 70; i++) aoDemo.render(64);
                const giFrame70 = measure(() => aoDemo.render(64), 20);
                for (let i = 0; i < 65; i++) aoDemo.render(64);
                const giSettled = measure(() => aoDemo.render(64), 25);
                let sweepDeg = 10;
                const giMovingSun = measure(() => {
                    sweepDeg = sweepDeg >= 40 ? 10 : sweepDeg + 2;
                    aoDemo.setSunAngle(sweepDeg);
                    aoDemo.render(64);
                }, 15);
                aoDemo.setSunAngle(22);
                const giEarly = measure(() => {
                    aoDemo.traced.frame.value = 0; aoDemo.resetHistory(); aoDemo.render(64);
                }, 15);

                return { aoEarly, aoSettled, giEarly, giFrame70, giSettled, giMovingSun };
            }""")

            browser.close()
        srv.shutdown()

    summary = {
        "bvh": bvh_meta,
        "radiance_grid": {
            **rg_meta,
            "relight_median_ms": float(np.median(check_res["relightTimes"])),
        },
        "accuracy": {
            "ao_mae_64": ao_mae,
            "bounce_irr_mae_64": bounce_mae,
            "gallery_ao_only_vs_gi_ref": {
                "rgb_mae": ao_only_mae_vs_gi_ref,
                "psnr_db": ao_only_psnr_vs_gi_ref,
            },
            "gallery_1bounce_gi_vs_gi_ref": {
                "rgb_mae": gi_mae_vs_gi_ref,
                "psnr_db": gi_psnr_vs_gi_ref,
            },
        },
        "timings_ms": timings,
        "errors": {
            "gl_errors": len(gl_errors),
            "console_errors": len(console_errors),
            "page_errors": len(page_errors),
            "gl_details": gl_errors,
            "console_details": console_errors,
            "page_details": page_errors,
        },
    }
    out_json = OUT_DIR / "gallery_gi_results.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== Gallery 1-Bounce GI Benchmark (Apple M1 Chrome WebGL2) ===")
    print(f"AO MAE@64:                  {ao_mae:.5f}")
    print(f"Bounce-only Irradiance MAE: {bounce_mae:.5f} (linear RGB)")
    print(f"Gallery AO-only  vs GI Ref: RGB MAE = {ao_only_mae_vs_gi_ref:.5f}, PSNR = {ao_only_psnr_vs_gi_ref:.2f} dB")
    print(f"Gallery 1-Bounce vs GI Ref: RGB MAE = {gi_mae_vs_gi_ref:.5f}, PSNR = {gi_psnr_vs_gi_ref:.2f} dB")
    print(f"Timing (AO-only):           early {timings['aoEarly']:.2f} ms/f, settled {timings['aoSettled']:.2f} ms/f")
    print(f"Timing (1-Bounce GI):       early {timings['giEarly']:.2f} ms/f, frame-70 {timings['giFrame70']:.2f} ms/f, settled {timings['giSettled']:.2f} ms/f, moving-sun {timings['giMovingSun']:.2f} ms/f")
    print(f"Saved report: {out_json}")

    if gl_errors or console_errors or page_errors:
        print("ERRORS DETECTED:", summary["errors"])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
