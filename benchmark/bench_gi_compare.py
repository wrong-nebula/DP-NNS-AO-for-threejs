#!/usr/bin/env python3
"""Apples-to-apples 1-Bounce Diffuse GI benchmark on Apple M1 (Gallery, 1280x720).

Evaluates pure 1-bounce diffuse GI irradiance (linear giIrr + tone-mapped sRGB PSNR)
against an independent 512-ray exact path-traced reference across TWO views:
  - View A (Front): all colored walls visible (best case for SSGI)
  - View B (Panned Right): terracotta crimson left wall off-screen (tests 360-deg off-screen bounce)

Compares:
  1. AO-only (no GI baseline)
  2. Ideal 512-ray SSGI (noiseless theoretical upper bound: exact frustum + depth-buffer visibility!)
  3. DDGI 8x8x5 World Irradiance Probe Grid (1 m spacing, 6-face ambient cube, exact 512-ray probes)
  4. Voxel Cone GI (VXGI, 5-cone mipmapped 64x64x40 radianceVol)   [measured live on Apple M1 GPU]
  5. Fast-SDF-GI WebGL2 (medium / high / ultra)                     [measured live on Apple M1 GPU]
  6. Fast-SDF-GI Unified WebGPU + SHaRC (low / medium / high / ultra) [measured live on M1 Metal-3]
"""
import functools
import http.server
import json
import math
import socketserver
import subprocess
import tempfile
import threading
from pathlib import Path
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "nextgen"
OUT.mkdir(parents=True, exist_ok=True)
TRACER_BIN = OUT / "bin" / "trace_gi"
W, H = 1280, 720


def srgb_to_lin(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, np.power((c + 0.055) / 1.055, 2.4))


def lin_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(np.maximum(c, 0.0), 1.0 / 2.4) - 0.055)


def aces(x):
    x = np.maximum(x, 0.0)
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)


def inv_aces(y):
    y = np.clip(y, 0.0, 0.996)
    a = 2.51 - 2.43 * y
    b = 0.03 - 0.59 * y
    c = -0.14 * y
    disc = np.maximum(b * b - 4.0 * a * c, 0.0)
    return np.maximum((-b + np.sqrt(disc)) / np.maximum(2.0 * a, 1e-6), 0.0)


def psnr(pred, ref):
    mse = float(np.mean((pred - ref) ** 2))
    return 99.0 if mse <= 1e-12 else float(10.0 * math.log10(1.0 / mse))


def score_gi(pred_lin, ref_lin, valid):
    """Score predicted linear bounce GI irradiance against 512-ray exact reference."""
    p = np.maximum(pred_lin[valid], 0.0)
    r = np.maximum(ref_lin[valid], 0.0)
    lin_mae = float(np.abs(p - r).mean())
    lin_rmse = float(np.sqrt(np.mean((p - r) ** 2)))
    # Perceptual tone-mapped sRGB bounce PSNR (same aces(gi * 1.15) view)
    p_srgb = lin_to_srgb(aces(p * 1.15))
    r_srgb = lin_to_srgb(aces(r * 1.15))
    srgb_mae = float(np.abs(p_srgb - r_srgb).mean())
    srgb_psnr = psnr(p_srgb, r_srgb)
    # Chromatic bleed energy ratio: mean(R - B) on pixels receiving warm red bleed (ref R - B > 0.06)
    red_mask = (r[:, 0] - r[:, 2]) > 0.06
    if np.any(red_mask):
        ref_bleed = float(np.mean(r[red_mask, 0] - r[red_mask, 2]))
        pred_bleed = float(np.mean(p[red_mask, 0] - p[red_mask, 2]))
        bleed_pct = round(100.0 * pred_bleed / max(ref_bleed, 1e-5), 1)
    else:
        bleed_pct = 100.0
    return {
        "lin_mae": round(lin_mae, 5),
        "lin_rmse": round(lin_rmse, 5),
        "srgb_mae": round(srgb_mae, 5),
        "srgb_psnr": round(srgb_psnr, 2),
        "red_bleed_pct": bleed_pct,
    }


def build_ssgi_bound_tracer(tmp):
    """Compile a 1-file C++ ideal 512-ray SSGI upper-bound tracer from tools/trace_gi.cpp."""
    src = (ROOT / "tools" / "trace_gi.cpp").read_text()
    # Inject camera frustum + occlusion visibility check on hitP to compute exact noiseless SSGI upper bound!
    ssgi_src = src.replace(
        "if (argc != 7) {",
        "if (argc != 17) {",
    ).replace(
        'std::fprintf(stderr, "usage: trace_gi gi_tris.f32 samples.f32 out_ref.f32 sx sy sz\\n");',
        'std::fprintf(stderr, "usage: trace_ssgi tris samp out sx sy sz ex ey ez tx ty tz fov aspect\\n");',
    ).replace(
        "const float GI_R = 8.5f;",
        """const float GI_R = 8.5f;
    V eye{std::stof(argv[7]), std::stof(argv[8]), std::stof(argv[9])};
    V tgt{std::stof(argv[10]), std::stof(argv[11]), std::stof(argv[12])};
    float fovRad = std::stof(argv[13]) * 3.14159265f / 180.f;
    float aspect = std::stof(argv[14]);
    float tanV = std::tan(fovRad * 0.5f), tanH = tanV * aspect;
    V fwd = norm(sub(tgt, eye));
    V right = norm(cross(fwd, V{0, 0, 1}));
    V up = cross(right, fwd);
    int ssgiMode = std::stoi(argv[15]); // 1 = ideal SSGI (only visible screen hits)
    (void)argv[16];""",
    ).replace(
        "giSum = add(giSum, mul3(ht.alb, irr));",
        """bool visOnScreen = true;
                            if (ssgiMode) {
                                V toHit = sub(hitP, eye);
                                float cz = dot(toHit, fwd);
                                float cx = dot(toHit, right);
                                float cy = dot(toHit, up);
                                if (cz <= 0.1f || std::fabs(cx) > cz * tanH || std::fabs(cy) > cz * tanV) {
                                    visOnScreen = false;
                                } else {
                                    float distCam = std::sqrt(dot(toHit, toHit));
                                    V toCam = mul(toHit, -1.f / std::max(distCam, 1e-4f));
                                    if (dot(hn, toCam) <= 0.02f) {
                                        visOnScreen = false; // back-facing to camera (not in G-buffer)
                                    } else if (bvh.any(add(hitP, mul(toCam, 0.01f)), toCam, distCam - 0.03f)) {
                                        visOnScreen = false; // occluded behind another surface in G-buffer
                                    }
                                }
                            }
                            if (visOnScreen) giSum = add(giSum, mul3(ht.alb, irr));""",
    )
    cpp_path = Path(tmp) / "trace_ssgi.cpp"
    bin_path = Path(tmp) / "trace_ssgi"
    cpp_path.write_text(ssgi_src)
    cmd = ["c++", "-O3", "-std=c++17", "-pthread"]
    try:
        sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
        inc = Path(sdk) / "usr/include/c++/v1"
        if inc.exists():
            cmd += ["-isystem", str(inc)]
    except Exception:
        pass
    cmd += [str(cpp_path), "-o", str(bin_path)]
    subprocess.check_call(cmd)
    return bin_path


def start_server():
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

    h = functools.partial(QuietHandler, directory=str(ROOT))
    socketserver.TCPServer.allow_reuse_address = True
    srv = socketserver.TCPServer(("127.0.0.1", 0), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def capture_gbuffer_and_trace_refs(page, gi_tris_path, ssgi_bin, tmp, eye, target, tag):
    page.evaluate("""({eye, target}) => {
      const d = aoDemo;
      d.renderer.setAnimationLoop(null);
      d.camera.position.set(...eye);
      d.controls.target.set(...target);
      d.controls.update();
      d.camera.updateMatrixWorld();
      d.setSunAngle(22);
      d.setGI(true);
      d.setMultiBounceGI(false);
      d.setEmissive(false);
      d.setMask(false);
      d.setGIMask(false);
      d.render(64);
    }""", {"eye": eye, "target": target})

    gb = page.evaluate("""() => {
      const d = aoDemo, T = d.T, r = d.renderer;
      const w = 320, h = 180;
      const mkRT = () => new T.WebGLRenderTarget(w, h, { type: T.FloatType, minFilter: T.NearestFilter, magFilter: T.NearestFilter });
      const posRT = mkRT(), norRT = mkRT(), albRT = mkRT();
      const mkMat = (expr) => new T.ShaderMaterial({
        glslVersion: T.GLSL3,
        vertexShader: `
          in vec3 surfAlbedo;
          out vec3 wp; out vec3 wn; out vec3 wa;
          void main() {
            vec4 wpos = modelMatrix * vec4(position, 1.);
            wp = wpos.xyz;
            wn = normalize(mat3(modelMatrix) * normal);
            wa = surfAlbedo;
            gl_Position = projectionMatrix * viewMatrix * wpos;
          }`,
        fragmentShader: `
          precision highp float;
          in vec3 wp; in vec3 wn; in vec3 wa;
          out vec4 o;
          void main() { o = ${expr}; }`,
      });
      const posMat = mkMat('vec4(wp, 1.)');
      const norMat = mkMat('vec4(normalize(wn), 1.)');
      const albMat = mkMat('vec4(wa, 1.)');
      const gm = d.galleryMesh, prevMat = gm.material, bg = d.scene.background;
      d.scene.background = null; r.setClearColor(0, 0);
      const pBuf = new Float32Array(w * h * 4);
      const nBuf = new Float32Array(w * h * 4);
      const aBuf = new Float32Array(w * h * 4);
      gm.material = posMat; r.setRenderTarget(posRT); r.render(d.scene, d.camera); r.readRenderTargetPixels(posRT, 0, 0, w, h, pBuf);
      gm.material = norMat; r.setRenderTarget(norRT); r.render(d.scene, d.camera); r.readRenderTargetPixels(norRT, 0, 0, w, h, nBuf);
      gm.material = albMat; r.setRenderTarget(albRT); r.render(d.scene, d.camera); r.readRenderTargetPixels(albRT, 0, 0, w, h, aBuf);
      r.setRenderTarget(null); gm.material = prevMat; d.scene.background = bg;
      posRT.dispose(); norRT.dispose(); albRT.dispose();
      posMat.dispose(); norMat.dispose(); albMat.dispose();
      return { p: Array.from(pBuf), n: Array.from(nBuf), a: Array.from(aBuf), sun: d.sunDir, fov: d.camera.fov };
    }""")

    p = np.array(gb["p"], dtype=np.float32).reshape(180, 320, 4)
    n = np.array(gb["n"], dtype=np.float32).reshape(180, 320, 4)
    a = np.array(gb["a"], dtype=np.float32).reshape(180, 320, 4)
    valid = p[:, :, 3] > 0.5
    n_len = np.linalg.norm(n[valid, :3], axis=-1)
    assert float(np.abs(n_len - 1.0).max()) < 0.02, f"Bad G-buffer normals: max err {np.abs(n_len - 1.0).max()}"
    samples = np.concatenate([p[:, :, :3], n[:, :, :3], a[:, :, :3]], axis=-1).reshape(-1, 9)
    samp_path = Path(tmp) / f"samples_{tag}.f32"
    ref_path = Path(tmp) / f"ref_{tag}.f32"
    ssgi_path = Path(tmp) / f"ssgi_{tag}.f32"
    samples.astype(np.float32).tofile(samp_path)
    sx, sy, sz = gb["sun"]
    subprocess.check_call([str(TRACER_BIN), str(gi_tris_path), str(samp_path), str(ref_path), str(sx), str(sy), str(sz)])
    subprocess.check_call([
        str(ssgi_bin), str(gi_tris_path), str(samp_path), str(ssgi_path),
        str(sx), str(sy), str(sz),
        str(eye[0]), str(eye[1]), str(eye[2]),
        str(target[0]), str(target[1]), str(target[2]),
        str(gb["fov"]), str(W / H), "1", "0",
    ])
    ref = np.fromfile(ref_path, dtype=np.float32).reshape(180, 320, 10)
    ssgi = np.fromfile(ssgi_path, dtype=np.float32).reshape(180, 320, 10)
    return {
        "valid": valid,
        "pos": p[:, :, :3],
        "nor": n[:, :, :3],
        "gi_ref": ref[:, :, 1:4],
        "ssgi_bound": ssgi[:, :, 1:4],
        "sun": np.array([sx, sy, sz], dtype=np.float32),
    }


def eval_ddgi_probe_grid(gi_tris_path, tmp, ref_view):
    """Evaluate 8x8x5 (1 m spacing) DDGI world irradiance probe grid (exact 512-ray 6-face probes)."""
    lo = np.array([-3.5, -3.1, 0.2], dtype=np.float32)
    hi = np.array([3.5, 3.5, 4.4], dtype=np.float32)
    nx, ny, nz = 8, 8, 5
    xs = np.linspace(lo[0], hi[0], nx)
    ys = np.linspace(lo[1], hi[1], ny)
    zs = np.linspace(lo[2], hi[2], nz)
    dirs = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=np.float32)
    pts, nors = [], []
    for z in zs:
        for y in ys:
            for x in xs:
                for d in dirs:
                    pts.append([x, y, z])
                    nors.append(d)
    pts = np.array(pts, dtype=np.float32)
    nors = np.array(nors, dtype=np.float32)
    p_samp = np.concatenate([pts, nors, np.ones_like(pts)], axis=-1)
    sp = Path(tmp) / "ddgi_probes.f32"
    rp = Path(tmp) / "ddgi_out.f32"
    if not rp.exists():
        p_samp.tofile(sp)
        sx, sy, sz = ref_view["sun"]
        subprocess.check_call([str(TRACER_BIN), str(gi_tris_path), str(sp), str(rp), str(sx), str(sy), str(sz)])
    probe_irr = np.fromfile(rp, dtype=np.float32).reshape(nz, ny, nx, 6, 10)[:, :, :, :, 1:4]

    out = np.zeros_like(ref_view["gi_ref"])
    v = ref_view["valid"]
    pos = ref_view["pos"][v]
    nor = ref_view["nor"][v]
    u = np.clip((pos - lo) / (hi - lo), 0.0, 1.0)
    gx = u[:, 0] * (nx - 1); ix = np.minimum(np.floor(gx).astype(int), nx - 2); fx = (gx - ix)[:, None]
    gy = u[:, 1] * (ny - 1); iy = np.minimum(np.floor(gy).astype(int), ny - 2); fy = (gy - iy)[:, None]
    gz = u[:, 2] * (nz - 1); iz = np.minimum(np.floor(gz).astype(int), nz - 2); fz = (gz - iz)[:, None]
    n2 = nor * nor
    dw = np.concatenate([
        np.where(nor[:, 0:1] >= 0, n2[:, 0:1], 0.0), np.where(nor[:, 0:1] < 0, n2[:, 0:1], 0.0),
        np.where(nor[:, 1:2] >= 0, n2[:, 1:2], 0.0), np.where(nor[:, 1:2] < 0, n2[:, 1:2], 0.0),
        np.where(nor[:, 2:3] >= 0, n2[:, 2:3], 0.0), np.where(nor[:, 2:3] < 0, n2[:, 2:3], 0.0),
    ], axis=-1)[:, :, None]

    ddgi_irr = np.zeros((len(pos), 3), dtype=np.float32)
    for dz in (0, 1):
        wz = fz if dz else (1.0 - fz)
        for dy in (0, 1):
            wy = fy if dy else (1.0 - fy)
            for dx in (0, 1):
                wx = fx if dx else (1.0 - fx)
                ddgi_irr += (wx * wy * wz) * np.sum(probe_irr[iz + dz, iy + dy, ix + dx] * dw, axis=1)
    out[v] = ddgi_irr
    return out


def read_webgl_stride4_srgb(page):
    arr = page.evaluate("""() => {
      const g = aoDemo.renderer.getContext();
      const px = new Uint8Array(1280 * 720 * 4);
      g.readPixels(0, 0, 1280, 720, g.RGBA, g.UNSIGNED_BYTE, px);
      const out = new Uint8Array(320 * 180 * 3);
      let k = 0;
      for (let j = 0; j < 180; j++) {
        const y = j * 4 + 2;
        for (let i = 0; i < 320; i++) {
          const x = i * 4 + 2;
          const o = (y * 1280 + x) * 4;
          out[k++] = px[o]; out[k++] = px[o+1]; out[k++] = px[o+2];
        }
      }
      return Array.from(out);
    }""")
    return np.array(arr, dtype=np.float32).reshape(180, 320, 3) / 255.0


def main():
    srv, port = start_server()
    views = {
        "front":  ([0.0, -5.95, 2.15], [0.0, 0.75, 1.55]),
        "panned": ([-0.9, -4.60, 2.05], [1.9, 1.25, 1.45]),  # red left wall off-screen!
    }
    results = []

    try:
        with tempfile.TemporaryDirectory(prefix="gi-cmp-") as tmp:
            ssgi_bin = build_ssgi_bound_tracer(tmp)
            gi_tris_path = Path(tmp) / "gallery_gi_tris.f32"
            subprocess.check_call(
                ["node", "tools/gen_gallery.mjs", "--tris-only", "--export-gi-tris", str(gi_tris_path)],
                cwd=str(ROOT),
            )

            with sync_playwright() as p:
                browser = p.chromium.launch(
                    channel="chrome",
                    headless=True,
                    args=[
                        "--enable-unsafe-webgpu",
                        "--use-gl=angle",
                        "--use-angle=metal",
                        "--enable-webgl",
                        "--ignore-gpu-blocklist",
                        "--enable-dawn-features=allow_unsafe_apis",
                    ],
                )
                page = browser.new_page(viewport={"width": 1280, "height": 720})
                page.goto(f"http://127.0.0.1:{port}/index.html", wait_until="networkidle")
                page.wait_for_function("() => window.aoDemo !== undefined", timeout=25000)
                page.click('button[data-scene="gallery"]')
                page.wait_for_timeout(1200)

                refs = {}
                for vname, (eye, tgt) in views.items():
                    refs[vname] = capture_gbuffer_and_trace_refs(page, gi_tris_path, ssgi_bin, tmp, eye, tgt, vname)

                # 1) Baseline: AO-only (0 bounce GI)
                row_ao = {"method": "AO-only (no GI)", "api": "Baseline"}
                for vname in views:
                    rv = refs[vname]
                    row_ao[vname] = score_gi(np.zeros_like(rv["gi_ref"]), rv["gi_ref"], rv["valid"])
                results.append(row_ao)

                # 2) Ideal 512-ray SSGI (exact theoretical noiseless upper bound)
                row_ssgi = {"method": "Ideal 512-ray SSGI (noiseless bound)", "api": "Screen-Space Bound"}
                for vname in views:
                    rv = refs[vname]
                    row_ssgi[vname] = score_gi(rv["ssgi_bound"], rv["gi_ref"], rv["valid"])
                results.append(row_ssgi)

                # 3) DDGI 8x8x5 World Probe Grid (1 m spacing, 6-face ambient cube, exact 512-ray probes)
                row_ddgi = {"method": "DDGI 8x8x5 Probe Grid (1 m, 512-ray)", "api": "World Probes"}
                for vname in views:
                    rv = refs[vname]
                    ddgi_lin = eval_ddgi_probe_grid(gi_tris_path, tmp, rv)
                    row_ddgi[vname] = score_gi(ddgi_lin, rv["gi_ref"], rv["valid"])
                results.append(row_ddgi)

                # 4) WebGL2 GPU methods (VXGI 5-cone low + Fast-SDF-GI medium/high/ultra) — pure bounce-only GI!
                for label, tier in [
                    ("Voxel Cone GI (VXGI 5-cone, 64^3)", "low"),
                    ("Fast-SDF-GI WebGL2 (medium)",       "medium"),
                    ("Fast-SDF-GI WebGL2 (high)",         "high"),
                    ("Fast-SDF-GI WebGL2 (ultra)",        "ultra"),
                ]:
                    row = {"method": label, "api": "WebGL2"}
                    for vname, (eye, tgt) in views.items():
                        page.evaluate("""({eye, target, tier}) => {
                          const d = aoDemo;
                          d.renderer.setAnimationLoop(null);
                          d.camera.position.set(...eye);
                          d.controls.target.set(...target);
                          d.controls.update();
                          d.camera.updateMatrixWorld();
                          d.setQuality(tier);
                          d.setGI(true);
                          d.setMultiBounceGI(false);
                          d.setEmissive(false);
                          d.setMask(false);
                          d.setGIMask(true);
                          d.setSunAngle(22);
                          d.resetHistory();
                          d.traced.frame.value = 0;
                          for (let f = 0; f < 96; f++) d.render(64);
                        }""", {"eye": eye, "target": tgt, "tier": tier})
                        # index.html giMask outputs linearToSrgb(clamp(giIrr, 0, 1))
                        pred_lin = srgb_to_lin(read_webgl_stride4_srgb(page))
                        rv = refs[vname]
                        row[vname] = score_gi(pred_lin, rv["gi_ref"], rv["valid"])
                        if vname == "front":
                            t = page.evaluate("""() => {
                              const d = aoDemo, g = d.renderer.getContext(), one = new Uint8Array(4);
                              d.setGIMask(false);
                              const early = [], settled = [];
                              for (let r = 0; r < 5; r++) {
                                d.resetHistory(); d.traced.frame.value = 0;
                                g.readPixels(0,0,1,1,g.RGBA,g.UNSIGNED_BYTE,one);
                                let t0 = performance.now();
                                for (let f = 0; f < 16; f++) d.render(64);
                                g.readPixels(0,0,1,1,g.RGBA,g.UNSIGNED_BYTE,one);
                                early.push((performance.now() - t0) / 16);
                              }
                              for (let f = 0; f < 130; f++) d.render(64);
                              for (let r = 0; r < 5; r++) {
                                g.readPixels(0,0,1,1,g.RGBA,g.UNSIGNED_BYTE,one);
                                let t0 = performance.now();
                                for (let f = 0; f < 24; f++) d.render(64);
                                g.readPixels(0,0,1,1,g.RGBA,g.UNSIGNED_BYTE,one);
                                settled.push((performance.now() - t0) / 24);
                              }
                              early.sort((a,b)=>a-b); settled.sort((a,b)=>a-b);
                              return { early_ms: +early[0].toFixed(2), settled_ms: +settled[0].toFixed(2) };
                            }""")
                            row.update(t)
                    results.append(row)

                # 5) Unified WebGPU Compute + SHaRC (webgpu.html) — pure bounce-only GI (viewMode == 2)!
                page.goto(f"http://127.0.0.1:{port}/webgpu.html", wait_until="networkidle")
                page.wait_for_function("() => window.wgpuDemo !== undefined", timeout=25000)
                page.evaluate("""() => {
                  const main = document.querySelector('main');
                  main.style.padding = '0'; main.style.maxWidth = 'none';
                  for (const el of main.children) if (el.id !== 'view') el.style.display = 'none';
                  document.querySelector('#hud').style.display = 'none';
                  const c = document.querySelector('#c');
                  c.style.width = '1280px'; c.style.height = '720px';
                }""")

                for label, tier in [
                    ("Fast-SDF-GI WebGPU+SHaRC (low)",    "low"),
                    ("Fast-SDF-GI WebGPU+SHaRC (medium)", "medium"),
                    ("Fast-SDF-GI WebGPU+SHaRC (high)",   "high"),
                    ("Fast-SDF-GI WebGPU+SHaRC (ultra)",  "ultra"),
                ]:
                    row = {"method": label, "api": "WebGPU"}
                    for vname, (eye, tgt) in views.items():
                        page.evaluate("""async ({eye, target, tier, sun}) => {
                          const d = wgpuDemo;
                          d.stopLoop();
                          await d.selectScene('gallery');
                          d.gpu.setQuality(tier);
                          d.gpu.useReflections = false;
                          d.gpu.viewMode = 2; // GI-only: linearToSrgb(acesToneMap(gi * 1.15))
                          d.setEmissive(false);
                          d.WEBGPU_QUALITY_PRESETS[tier].multiBounce = false;
                          d.gpu.sunDir = sun;
                          d.gpu.relight(false);
                          d.gpu.clearHash();
                          d.camera.position.set(...eye);
                          d.controls.target.set(...target);
                          d.controls.update();
                          for (let f = 0; f < 96; f++) d.renderOnce();
                          await d.gpu.device.queue.onSubmittedWorkDone();
                        }""", {"eye": eye, "target": tgt, "tier": tier, "sun": refs[vname]["sun"].tolist()})
                        tmp_png = Path(tmp) / f"wgpu_{tier}_{vname}.png"
                        page.locator("#c").screenshot(path=str(tmp_png))
                        im_srgb = np.array(Image.open(tmp_png).convert("RGB"), dtype=np.float32)[::-1, :, :] / 255.0
                        pred_srgb = im_srgb[2::4, 2::4, :]
                        pred_lin = inv_aces(srgb_to_lin(pred_srgb)) / 1.15
                        rv = refs[vname]
                        row[vname] = score_gi(pred_lin, rv["gi_ref"], rv["valid"])
                        if vname == "front":
                            t = page.evaluate("""async () => {
                              const d = wgpuDemo;
                              d.gpu.viewMode = 0;
                              const early = [], settled = [];
                              for (let r = 0; r < 5; r++) {
                                d.gpu.clearHash();
                                let t0 = performance.now();
                                for (let f = 0; f < 16; f++) d.renderOnce();
                                await d.gpu.device.queue.onSubmittedWorkDone();
                                early.push((performance.now() - t0) / 16);
                              }
                              for (let f = 0; f < 96; f++) d.renderOnce();
                              await d.gpu.device.queue.onSubmittedWorkDone();
                              for (let r = 0; r < 5; r++) {
                                let t0 = performance.now();
                                for (let f = 0; f < 24; f++) d.renderOnce();
                                await d.gpu.device.queue.onSubmittedWorkDone();
                                settled.push((performance.now() - t0) / 24);
                              }
                              early.sort((a,b)=>a-b); settled.sort((a,b)=>a-b);
                              return { early_ms: +early[0].toFixed(2), settled_ms: +settled[0].toFixed(2), gpu_ms: +d.gpu.lastGpuMs.toFixed(2) };
                            }""")
                            row.update(t)
                    results.append(row)

                browser.close()
    finally:
        srv.shutdown()

    (OUT / "gi_compare_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
