#!/usr/bin/env python3
"""Anti-Goodhart Perceptual, Temporal & Adversarial Judge v2 for Fast-SDF-AO.

Runs the unmodified production index.html on real Apple M1 ANGLE/Metal GPU via
Playwright Chrome and scores:
  1. 6-view spatial, contact, tail-risk (CVaR-5%), Oklab L*, FLIP-style perceptual
     error, and GMSD edge error at frames [1, 4, 16, 64] + auto-cropped top-5
     worst-error board & FLIP heatmap PNGs.
  2. Camera-cut convergence curve (AUC, disocclusion frames 1..4), 32-frame 3 deg/f
     orbit + settle, and static temporal flicker.
  3. Frozen & continuous-motion mover accuracy (detects mover lag/ghosting).
  4. Adversarial thin-plate (2/4/8/12/20 cm) + curved/organic (sphere+torus-knot)
     geometry probes against exact ray-traced references.
  5. Real GPU timing (wall + EXT_disjoint_timer_query_webgl2), startup build ms,
     VRAM bytes, and 7-mode zero-GL-error check.

Usage:
  python3 benchmark/judge_v2.py --tag baseline-main [--quick] [--url /index.html]
"""
import argparse
import functools
import http.server
import json
import math
import socketserver
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "output/playwright/traced-run2"
OUT_ROOT = ROOT / "output/nextgen"
TRACER_BIN = OUT_ROOT / "bin/trace_tris"

VIEWS_FIT = ["main", "west", "north"]
VIEWS_HELD = ["low", "inner", "above"]
ALL_VIEWS = VIEWS_FIT + VIEWS_HELD


# ---------------------------------------------------------------------------
# Pure-NumPy 2D Perceptual Metrics (Oklab L*, FLIP-style CSF+Edge, GMSD)
# ---------------------------------------------------------------------------
def gaussian_blur_2d(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Normalized convolution Gaussian blur respecting valid pixel mask."""
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()

    def conv1d(arr, axis):
        pad_width = [(0, 0), (0, 0)]
        pad_width[axis] = (radius, radius)
        padded = np.pad(arr, pad_width, mode="edge")
        out = np.zeros_like(arr, dtype=np.float64)
        for i, w in enumerate(k):
            if axis == 0:
                out += w * padded[i : i + arr.shape[0], :]
            else:
                out += w * padded[:, i : i + arr.shape[1]]
        return out

    num = conv1d(conv1d(img * mask, 0), 1)
    den = conv1d(conv1d(mask.astype(np.float64), 0), 1)
    return np.where(den > 1e-8, num / np.maximum(den, 1e-8), 0.0)


def sobel_mag(img: np.ndarray) -> np.ndarray:
    p = np.pad(img, 1, mode="edge")
    gx = (
        -p[:-2, :-2]
        + p[:-2, 2:]
        - 2.0 * p[1:-1, :-2]
        + 2.0 * p[1:-1, 2:]
        - p[2:, :-2]
        + p[2:, 2:]
    ) * 0.25
    gy = (
        -p[:-2, :-2]
        - 2.0 * p[:-2, 1:-1]
        - p[:-2, 2:]
        + p[2:, :-2]
        + 2.0 * p[2:, 1:-1]
        + p[2:, 2:]
    ) * 0.25
    return np.sqrt(gx * gx + gy * gy + 1e-12)


def compute_2d_perceptual(pred_1d: np.ndarray, ref_1d: np.ndarray, indices: np.ndarray):
    """Map 4-stride samples back to 180x320 grid and compute Oklab, FLIP-style, GMSD."""
    H_s, W_s = 180, 320
    pred_img = np.zeros((H_s, W_s), dtype=np.float64)
    ref_img = np.zeros((H_s, W_s), dtype=np.float64)
    valid = np.zeros((H_s, W_s), dtype=bool)

    # indices = (719 - y) * 1280 + x where x = 2 + 4*ix, y = 2 + 4*iy
    ys = 719 - (indices // 1280)
    xs = indices % 1280
    iy = (ys - 2) // 4
    ix = (xs - 2) // 4
    pred_img[iy, ix] = np.clip(pred_1d, 0.0, 1.0)
    ref_img[iy, ix] = np.clip(ref_1d, 0.0, 1.0)
    valid[iy, ix] = True

    # Oklab lightness L* = cbrt(Y) for achromatic linear AO
    L_pred = np.cbrt(np.clip(pred_img, 0.0, 1.0))
    L_ref = np.cbrt(np.clip(ref_img, 0.0, 1.0))
    oklab_mae = float(np.mean(np.abs(L_pred[valid] - L_ref[valid])))

    # Multi-scale CSF-filtered perceptual luminance difference (FLIP-style)
    # Approximate human achromatic CSF on 4-stride grid (1 pixel ~ 4 full-res px)
    csf_pred = 0.65 * gaussian_blur_2d(L_pred, valid, 0.55) + 0.35 * gaussian_blur_2d(L_pred, valid, 1.35)
    csf_ref = 0.65 * gaussian_blur_2d(L_ref, valid, 0.55) + 0.35 * gaussian_blur_2d(L_ref, valid, 1.35)
    dE = np.clip(np.abs(csf_pred - csf_ref) * 2.5, 0.0, 1.0)

    # Feature/edge difference via Sobel gradient magnitude on Oklab L*
    g_pred = sobel_mag(L_pred)
    g_ref = sobel_mag(L_ref)
    # Only score edges where 3x3 neighborhood is valid
    v_f = valid.astype(np.float64)
    v_neigh = (
        np.pad(v_f, 1, mode="constant")[0:-2, 0:-2]
        * np.pad(v_f, 1, mode="constant")[0:-2, 1:-1]
        * np.pad(v_f, 1, mode="constant")[0:-2, 2:]
        * np.pad(v_f, 1, mode="constant")[1:-1, 0:-2]
        * np.pad(v_f, 1, mode="constant")[1:-1, 1:-1]
        * np.pad(v_f, 1, mode="constant")[1:-1, 2:]
        * np.pad(v_f, 1, mode="constant")[2:, 0:-2]
        * np.pad(v_f, 1, mode="constant")[2:, 1:-1]
        * np.pad(v_f, 1, mode="constant")[2:, 2:]
    ) > 0.5

    dF = np.clip(np.abs(g_pred - g_ref) / (np.maximum(g_pred, g_ref) + 0.08), 0.0, 1.0)
    dF = np.where(v_neigh, dF, 0.0)

    # FLIP combination: emphasize feature error on edges + color/lightness error
    flip_map = np.where(valid, np.power(dE, 0.7) * (1.0 - dF) + dF * np.sqrt(np.maximum(dE, 0.05)), 0.0)
    flip_vals = flip_map[valid]

    # GMSD (Xue et al. 2014): std dev of Gradient Magnitude Similarity map
    T_gmsd = 0.0026
    gms = (2.0 * g_pred * g_ref + T_gmsd) / (g_pred * g_pred + g_ref * g_ref + T_gmsd)
    gmsd = float(np.std(gms[v_neigh])) if np.any(v_neigh) else 0.0

    return {
        "oklab_mae": oklab_mae,
        "flip_mean": float(np.mean(flip_vals)),
        "flip_p95": float(np.percentile(flip_vals, 95)),
        "flip_max": float(np.max(flip_vals)),
        "gmsd": gmsd,
        "flip_map": flip_map,
        "abs_err_img": np.where(valid, np.abs(pred_img - ref_img), 0.0),
        "pred_img": pred_img,
        "ref_img": ref_img,
        "valid": valid,
    }


def score_samples(pred: np.ndarray, ref: np.ndarray, contact: np.ndarray):
    err = pred - ref
    abs_err = np.abs(err)
    cmask = contact.astype(bool)
    k5 = max(1, int(math.ceil(0.05 * len(abs_err))))
    worst5 = np.partition(abs_err, len(abs_err) - k5)[len(abs_err) - k5 :]
    c_err = err[cmask]
    return {
        "mae": float(np.mean(abs_err)),
        "contact_mae": float(np.mean(np.abs(c_err))) if np.any(cmask) else 0.0,
        "under_occ": float(np.mean(np.maximum(0.0, c_err))) if np.any(cmask) else 0.0,
        "over_occ": float(np.mean(np.maximum(0.0, -c_err))) if np.any(cmask) else 0.0,
        "cvar5": float(np.mean(worst5)),
        "p99": float(np.percentile(abs_err, 99)),
    }


def save_flip_heatmap(flip_map: np.ndarray, out_path: Path):
    """Save magma-like colorized FLIP heatmap PNG."""
    v = np.clip(flip_map, 0.0, 1.0)
    # Simple perceptual magma-inspired ramp
    r = np.clip(1.6 * np.power(v, 0.65), 0.0, 1.0)
    g = np.clip(1.8 * np.power(np.maximum(v - 0.25, 0.0), 1.3), 0.0, 1.0)
    b = np.clip(0.55 * np.sin(v * math.pi) + 0.6 * np.power(np.maximum(v - 0.65, 0.0), 1.5), 0.0, 1.0)
    rgb = (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)
    Image.fromarray(rgb).resize((640, 360), Image.Resampling.NEAREST).save(out_path)


def save_top5_crops(pred_img: np.ndarray, ref_img: np.ndarray, abs_err_img: np.ndarray, out_path: Path):
    """Automatically find top-5 non-overlapping worst-error regions and save visual diff board."""
    H_s, W_s = abs_err_img.shape
    err_work = abs_err_img.copy()
    patch = 16  # 16x16 in 4-stride = 64x64 in full-res
    half = patch // 2
    regions = []
    for _ in range(5):
        idx = int(np.argmax(err_work))
        cy, cx = divmod(idx, W_s)
        val = float(err_work[cy, cx])
        if val <= 0:
            break
        regions.append((cy, cx, val))
        y0, y1 = max(0, cy - patch), min(H_s, cy + patch)
        x0, x1 = max(0, cx - patch), min(W_s, cx + patch)
        err_work[y0:y1, x0:x1] = -1.0

    scale = 6  # 16*6 = 96px tiles
    tile = patch * scale
    cols = 3  # Ref | Pred | 4x |Err|
    board = Image.new("RGB", (cols * tile + 12, len(regions) * (tile + 18) + 28), (18, 24, 33))
    draw = ImageDraw.Draw(board)
    draw.text((8, 6), "Top-5 Worst Error Regions: [Ref]  [Pred]  [4x |Err|]", fill=(230, 238, 245))

    for row, (cy, cx, val) in enumerate(regions):
        y0 = max(0, min(H_s - patch, cy - half))
        x0 = max(0, min(W_s - patch, cx - half))
        r_p = (np.clip(ref_img[y0 : y0 + patch, x0 : x0 + patch], 0, 1) * 255).astype(np.uint8)
        p_p = (np.clip(pred_img[y0 : y0 + patch, x0 : x0 + patch], 0, 1) * 255).astype(np.uint8)
        e_p = (np.clip(abs_err_img[y0 : y0 + patch, x0 : x0 + patch] * 4.0, 0, 1) * 255).astype(np.uint8)
        e_rgb = np.stack([e_p, (e_p * 0.35).astype(np.uint8), np.zeros_like(e_p)], axis=-1)

        oy = 26 + row * (tile + 18)
        draw.text((8, oy), f"#{row+1} full=({x0*4+2},{y0*4+2}) maxErr={val:.3f}", fill=(200, 212, 224))
        for col, arr in enumerate([np.stack([r_p] * 3, -1), np.stack([p_p] * 3, -1), e_rgb]):
            im = Image.fromarray(arr).resize((tile, tile), Image.Resampling.NEAREST)
            board.paste(im, (4 + col * (tile + 4), oy + 14))
    board.save(out_path)


# ---------------------------------------------------------------------------
# Curved / Organic Geometry Reference (Exact triangle ray-tracing via trace_tris)
# ---------------------------------------------------------------------------
def build_curved_fixture_ref(cache_path: Path):
    """Sphere (r=0.28 resting at z=0.28) + TorusKnot hovering 3 cm above floor."""
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    # Generate icosphere-like UV sphere triangles & torus-knot-like ring triangles in Python
    # Wait: even cleaner, generate the exact same geometry in JS (Three.js Sphere + Torus)
    # and pass its triangles to trace_tris, or use a simple analytic sphere + tilted cylinder/torus!
    # Here we build a pure-math closed UV sphere + closed torus + floor box so JS and Python match bit-for-bit.
    tris = []

    def add_quad(a, b, c, d):
        tris.append([a, b, c])
        tris.append([a, c, d])

    # Floor slab [-2,2]x[-2,2]x[-0.1,0]
    for z in [0.0, -0.1]:
        p0, p1, p2, p3 = [-2, -2, z], [2, -2, z], [2, 2, z], [-2, 2, z]
        if z == 0.0:
            add_quad(p0, p1, p2, p3)
        else:
            add_quad(p0, p3, p2, p1)

    # Sphere r=0.28 centered at (-0.32, 0.0, 0.28) resting on z=0
    scx, scy, scz, sr = -0.32, 0.0, 0.28, 0.28
    nu, nv = 24, 14
    for j in range(nv):
        t0 = math.pi * j / nv
        t1 = math.pi * (j + 1) / nv
        for i in range(nu):
            p0 = 2 * math.pi * i / nu
            p1 = 2 * math.pi * (i + 1) / nu
            def sp(th, ph):
                return [
                    scx + sr * math.sin(th) * math.cos(ph),
                    scy + sr * math.sin(th) * math.sin(ph),
                    scz + sr * math.cos(th),
                ]
            a, b, c, d = sp(t0, p0), sp(t1, p0), sp(t1, p1), sp(t0, p1)
            if j > 0:
                tris.append([a, b, d])
            if j < nv - 1:
                tris.append([b, c, d])

    # Horizontal Torus R=0.26, r=0.09 centered at (+0.34, 0.0, 0.12) -> 3 cm gap above floor!
    tcx, tcy, tcz, tR, tr = 0.34, 0.0, 0.12, 0.26, 0.09
    tu, tv = 32, 16
    for i in range(tu):
        u0 = 2 * math.pi * i / tu
        u1 = 2 * math.pi * (i + 1) / tu
        for j in range(tv):
            v0 = 2 * math.pi * j / tv
            v1 = 2 * math.pi * (j + 1) / tv
            def tp(u, v):
                rr = tR + tr * math.cos(v)
                return [tcx + rr * math.cos(u), tcy + rr * math.sin(u), tcz + tr * math.sin(v)]
            add_quad(tp(u0, v0), tp(u1, v0), tp(u1, v1), tp(u0, v1))

    tris_arr = np.array(tris, dtype=np.float32).reshape(-1, 9)
    xs = np.linspace(-0.8, 0.8, 129, dtype=np.float64)
    n_rays = 4096
    idx = np.arange(n_rays, dtype=np.float64)
    u = (idx + 0.5) / n_rays
    phi = idx * math.pi * (3.0 - math.sqrt(5.0))
    dirs = np.stack([np.sqrt(u) * np.cos(phi), np.sqrt(u) * np.sin(phi), np.sqrt(1.0 - u)], axis=-1)

    origins = np.zeros((len(xs), n_rays, 3), dtype=np.float32)
    origins[:, :, 0] = xs[:, None]
    origins[:, :, 2] = 0.0003
    ray_dirs = np.broadcast_to(dirs[None, :, :], (len(xs), n_rays, 3)).astype(np.float32)
    tmax = np.full((len(xs), n_rays, 1), 1.12, dtype=np.float32)
    rays = np.concatenate([origins, ray_dirs, tmax], axis=-1).reshape(-1, 7)

    tmp_dir = OUT_ROOT / "tmp_curved"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tris_arr.tofile(tmp_dir / "tris.f32")
    rays.tofile(tmp_dir / "rays.f32")
    subprocess.run(
        [str(TRACER_BIN), str(tmp_dir / "tris.f32"), str(tmp_dir / "rays.f32"), str(tmp_dir / "hits.u8")],
        check=True,
        capture_output=True,
    )
    hits = np.fromfile(tmp_dir / "hits.u8", dtype=np.uint8).reshape(len(xs), n_rays)
    ref_ao = (1.0 - hits.mean(axis=1)).tolist()
    data = {"xs": xs.tolist(), "ref": ref_ao, "tris": tris_arr.reshape(-1).tolist(), "triCount": len(tris_arr)}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data))
    return data


# ---------------------------------------------------------------------------
# Local HTTP Server
# ---------------------------------------------------------------------------
class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def start_server(root: Path):
    handler = functools.partial(QuietHandler, directory=str(root))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, port


# ---------------------------------------------------------------------------
# Main Judge Runner
# ---------------------------------------------------------------------------
def run_judge(tag: str, url_path: str, quick: bool = False):
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    curved_ref = build_curved_fixture_ref(OUT_ROOT / "curved_fixture_ref.json")

    httpd, port = start_server(ROOT)
    base_url = f"http://127.0.0.1:{port}{url_path}"
    console_errors = []
    page_errors = []

    t0_all = time.perf_counter()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                channel="chrome",
                headless=True,
                args=["--use-gl=angle", "--use-angle=metal", "--ignore-gpu-blocklist"],
            )
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
            page.on("pageerror", lambda err: page_errors.append(str(err)))

            page.goto(base_url, wait_until="domcontentloaded")
            page.wait_for_function("() => window.aoDemo !== undefined", timeout=60000)

            # 1. GPU info & 7-mode GL sanity check
            env_info = page.evaluate("""async () => {
                const d = aoDemo, gl = d.renderer.getContext();
                d.renderer.setAnimationLoop(null);
                d.controls.enableDamping = false;
                const dbg = gl.getExtension('WEBGL_debug_renderer_info');
                const renderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
                const vendor = dbg ? gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR);
                const timerExt = gl.getExtension('EXT_disjoint_timer_query_webgl2');
                // Clear any startup GL state
                while (gl.getError() !== gl.NO_ERROR) {}
                const modeErrors = {};
                for (const m of [0, 4, 14, 34, 32, 36, 64]) {
                    d.render(m);
                    modeErrors[m] = gl.getError();
                }
                const cr = d.contactResources;
                return {
                    vendor, renderer,
                    hasTimerQuery: !!timerExt,
                    modeErrors,
                    bakeSyncMs: d.bakeSyncMs,
                    contactBuildMs: cr ? cr.buildMs : null,
                    contactBytes: cr ? cr.bytes : null,
                    contactNodes: cr ? cr.nodes : null,
                    contactTriangles: cr ? cr.triangles : null,
                    contactRange: cr ? cr.range : null,
                };
            }""")
            if "Apple M1" not in env_info["renderer"] and "Metal" not in env_info["renderer"]:
                raise RuntimeError(f"Expected real Apple Metal GPU, got: {env_info['renderer']}")

            # 2. 6-view evaluation at [1, 4, 16, 64] + main convergence curve
            view_scores = {}
            main_curve = {}
            main_diag = {}
            flicker_static = None

            for vname in ALL_VIEWS:
                ref_data = json.loads((REF_DIR / f"{vname}.json").read_text())
                ref_arr = np.array(ref_data["ref"], dtype=np.float64)
                contact_arr = np.array(ref_data["contact"], dtype=np.int32)
                indices_arr = np.array(ref_data["indices"], dtype=np.int32)
                checkpoints = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64] if vname == "main" else [1, 4, 16, 64]
                collect_flicker = (vname == "main")

                res = page.evaluate("""({eye, target, indices, checkpoints, collectFlicker}) => {
                    const d = aoDemo, gl = d.renderer.getContext();
                    d.setDynamicAO(false);
                    d.movers.forEach(m => m.object.visible = false);
                    d.setAOScale(0.5);
                    d.setMask(true);
                    d.camera.position.fromArray(eye);
                    d.controls.target.fromArray(target);
                    d.controls.update();
                    d.camera.lookAt(...target);
                    d.camera.updateMatrixWorld();
                    d.traced.frame.value = 0;
                    d.resetHistory();
                    const buf = new Uint8Array(1280 * 720 * 4);
                    const out = {};
                    const diag = {};
                    let prevBuf = null, flickerSum = 0, flickerCnt = 0;
                    for (let f = 1; f <= 64; f++) {
                        d.render(64);
                        const want = checkpoints.includes(f);
                        const wantFlick = collectFlicker && f >= 56 && f <= 64;
                        if (want || wantFlick) {
                            gl.readPixels(0, 0, 1280, 720, gl.RGBA, gl.UNSIGNED_BYTE, buf);
                            if (want) {
                                const vals = new Array(indices.length);
                                for (let i = 0; i < indices.length; i++) vals[i] = buf[indices[i] * 4] / 255;
                                out[f] = vals;
                                if (f === 64 && collectFlicker) {
                                    diag.pix416854 = buf[416854 * 4] / 255;
                                    diag.pix575366 = buf[575366 * 4] / 255;
                                }
                            }
                            if (wantFlick) {
                                if (prevBuf) {
                                    let s = 0;
                                    for (let i = 0; i < indices.length; i++) {
                                        s += Math.abs(buf[indices[i] * 4] - prevBuf[indices[i] * 4]) / 255;
                                    }
                                    flickerSum += s / indices.length;
                                    flickerCnt++;
                                }
                                prevBuf = new Uint8Array(buf);
                            }
                        }
                    }
                    return { out, diag, flicker: flickerCnt ? flickerSum / flickerCnt : 0 };
                }""", {"eye": ref_data["eye"], "target": ref_data["target"], "indices": ref_data["indices"], "checkpoints": checkpoints, "collectFlicker": collect_flicker})

                view_scores[vname] = {}
                for f in [1, 4, 16, 64]:
                    pred = np.array(res["out"][str(f)], dtype=np.float64)
                    sc = score_samples(pred, ref_arr, contact_arr)
                    if f in (16, 64):
                        perc = compute_2d_perceptual(pred, ref_arr, indices_arr)
                        sc["oklab_mae"] = perc["oklab_mae"]
                        sc["flip_mean"] = perc["flip_mean"]
                        sc["flip_p95"] = perc["flip_p95"]
                        sc["flip_max"] = perc["flip_max"]
                        sc["gmsd"] = perc["gmsd"]
                        if vname == "main" and f == 64:
                            save_flip_heatmap(perc["flip_map"], out_dir / "main_flip_64.png")
                            save_top5_crops(perc["pred_img"], perc["ref_img"], perc["abs_err_img"], out_dir / "main_top5_crops.png")
                    view_scores[vname][str(f)] = sc

                if vname == "main":
                    for f in checkpoints:
                        pred = np.array(res["out"][str(f)], dtype=np.float64)
                        main_curve[f] = float(np.mean(np.abs(pred - ref_arr)))
                    main_diag = res["diag"]
                    flicker_static = res["flicker"]

            # Convergence AUC (normalized trapezoidal integral over frames 1..64) & disocclusion band (frames 1..4)
            xs_c = sorted(main_curve.keys())
            ys_c = [main_curve[k] for k in xs_c]
            conv_auc = float(np.trapezoid(ys_c, xs_c) / (xs_c[-1] - xs_c[0]))
            disocc_mae = float(np.mean([main_curve[1], main_curve[2], main_curve[3], main_curve[4]]))

            # 3. 32-frame 3 deg/frame orbit + settle ending at main
            main_ref = json.loads((REF_DIR / "main.json").read_text())
            main_ref_arr = np.array(main_ref["ref"], dtype=np.float64)
            orbit_res = page.evaluate("""({eye, target, indices}) => {
                const d = aoDemo, gl = d.renderer.getContext();
                d.setDynamicAO(false);
                d.movers.forEach(m => m.object.visible = false);
                d.setAOScale(0.5);
                d.setMask(true);
                d.traced.frame.value = 0;
                d.resetHistory();
                function setOrbit(deg) {
                    const ang = deg * Math.PI / 180;
                    const x = eye[0] - target[0], y = eye[1] - target[1];
                    d.camera.position.set(target[0] + x * Math.cos(ang) - y * Math.sin(ang),
                                          target[1] + x * Math.sin(ang) + y * Math.cos(ang), eye[2]);
                    d.camera.lookAt(...target);
                    d.camera.updateMatrixWorld();
                }
                const buf = new Uint8Array(1280 * 720 * 4);
                function sample() {
                    gl.readPixels(0, 0, 1280, 720, gl.RGBA, gl.UNSIGNED_BYTE, buf);
                    const vals = new Array(indices.length);
                    for (let i = 0; i < indices.length; i++) vals[i] = buf[indices[i] * 4] / 255;
                    return vals;
                }
                // 32 frames at 3 deg/frame ending at 0 deg (main)
                for (let i = 0; i < 32; i++) {
                    setOrbit((i - 31) * 3);
                    d.render(64);
                }
                const end0 = sample();
                // Settle 16 frames at main
                let s4 = null, s16 = null;
                for (let i = 1; i <= 16; i++) {
                    d.render(64);
                    if (i === 4) s4 = sample();
                    if (i === 16) s16 = sample();
                }
                return { end0, s4, s16 };
            }""", {"eye": main_ref["eye"], "target": main_ref["target"], "indices": main_ref["indices"]})
            orbit_metrics = {
                "orbit_3deg_end_mae": float(np.mean(np.abs(np.array(orbit_res["end0"]) - main_ref_arr))),
                "orbit_settle4_mae": float(np.mean(np.abs(np.array(orbit_res["s4"]) - main_ref_arr))),
                "orbit_settle16_mae": float(np.mean(np.abs(np.array(orbit_res["s16"]) - main_ref_arr))),
            }

            # 4. Frozen + Continuous Mover Accuracy (4 held-out poses)
            movers_ref = json.loads((REF_DIR / "movers.json").read_text())
            mover_eval = page.evaluate("""({eye, target, indices, times}) => {
                const d = aoDemo, gl = d.renderer.getContext();
                d.setAOScale(0.5);
                d.setMask(true);
                d.movers.forEach(m => m.object.visible = true);
                d.setDynamicAO(true);
                d.camera.position.fromArray(eye);
                d.controls.target.fromArray(target);
                d.controls.update();
                d.camera.lookAt(...target);
                d.camera.updateMatrixWorld();
                const buf = new Uint8Array(1280 * 720 * 4);
                function sample() {
                    gl.readPixels(0, 0, 1280, 720, gl.RGBA, gl.UNSIGNED_BYTE, buf);
                    const vals = new Array(indices.length);
                    for (let i = 0; i < indices.length; i++) vals[i] = buf[indices[i] * 4] / 255;
                    return vals;
                }
                const frozen = {}, continuous = {};
                for (const tStr of times) {
                    const t = parseFloat(tStr);
                    // Frozen: reset history, render 32 frames at fixed t
                    d.traced.frame.value = 0; d.resetHistory();
                    d.updateMovers(t);
                    for (let f = 0; f < 32; f++) d.render(64);
                    frozen[tStr] = sample();

                    // Continuous: settle 16 frames at t - 16*dt, then step 16 frames continuously to t
                    const dt = 0.04;
                    d.traced.frame.value = 0; d.resetHistory();
                    d.updateMovers(t - 16 * dt);
                    for (let f = 0; f < 16; f++) d.render(64);
                    for (let step = 1; step <= 16; step++) {
                        d.updateMovers(t - (16 - step) * dt);
                        d.render(64);
                    }
                    continuous[tStr] = sample();
                }
                return { frozen, continuous };
            }""", {"eye": main_ref["eye"], "target": main_ref["target"], "indices": main_ref["indices"], "times": list(movers_ref.keys())})

            def eval_movers_dict(pred_dict):
                all_maes, near_maes = [], []
                for t_str, ref_m in movers_ref.items():
                    pred = np.array(pred_dict[t_str], dtype=np.float64)
                    ao_r = np.array([np.nan if v is None else v for v in ref_m["ao"]], dtype=np.float64)
                    scored = np.array(ref_m["scored"], dtype=bool) & np.isfinite(ao_r)
                    near = scored & np.array(ref_m["affected"], dtype=bool)
                    all_maes.append(float(np.mean(np.abs(pred[scored] - ao_r[scored]))))
                    near_maes.append(float(np.mean(np.abs(pred[near] - ao_r[near]))))
                return {"all_mae": float(np.mean(all_maes)), "near_mae": float(np.mean(near_maes))}

            mover_metrics = {
                "frozen": eval_movers_dict(mover_eval["frozen"]),
                "continuous": eval_movers_dict(mover_eval["continuous"]),
            }

            # 5. Adversarial Thin-Plate & Curved Geometry Probes
            probes = page.evaluate("""async ({curvedTris, curvedXs}) => {
                const d = aoDemo, T = d.T, R = d.renderer;
                const bvhMod = await import('/contact-bvh.js');
                const buildBVH = bvhMod.buildContactBVH;
                const fragSrc = d.aoPass.material.fragmentShader;
                // Extract everything from CONTACT_GLSL / tracedAO helpers cleanly:
                // We wrap staticDistance + frame + uniforms and call tracedAO(p, n)
                const traceBody = fragSrc.slice(fragSrc.indexOf('float ign('), fragSrc.indexOf('// The previous frame'));
                const probeFrag = `precision highp float; precision highp sampler3D;
                uniform float frame; uniform float gap; uniform int probeMode;
                float staticDistance(vec3 p) {
                    if (probeMode == 0) {
                        vec3 q = abs(p - vec3(0., 0., gap + .01)) - vec3(.25, .25, .01);
                        return min(p.z, length(max(q, 0.)) + min(max(q.x, max(q.y, q.z)), 0.));
                    } else {
                        float dSphere = length(p - vec3(-0.32, 0.0, 0.28)) - 0.28;
                        vec3 qt = p - vec3(0.34, 0.0, 0.12);
                        float dTorus = length(vec2(length(qt.xy) - 0.26, qt.z)) - 0.09;
                        return min(p.z, min(dSphere, dTorus));
                    }
                }
                ${traceBody}
                out vec4 probeOut;
                void main() {
                    float x = (gl_FragCoord.x - 0.5) / 128.0 * (probeMode == 0 ? 0.8 : 1.6) - (probeMode == 0 ? 0.4 : 0.8);
                    probeOut = vec4(tracedAO(vec3(x, 0., 0.), vec3(0., 0., 1.)), 0., 0., 1.);
                }`;
                const W = 129, H = 16;
                const u = {
                    frame: {value: 0}, gap: {value: 0.02}, probeMode: {value: 0},
                    contactRange: {value: d.aoPass.material.uniforms.contactRange.value},
                    contactBias: {value: d.aoPass.material.uniforms.contactBias.value},
                };
                // Copy any extra uniforms (like STBN texture) if present on aoPass
                for (const [k, v] of Object.entries(d.aoPass.material.uniforms)) {
                    if (!(k in u) && !['field','gPos','gNormal','histTex','normalTex','prevViewProj','histMax','settleFrames','tileRate'].includes(k)) {
                        u[k] = { value: v.value };
                    }
                }
                const mat = new T.ShaderMaterial({
                    glslVersion: T.GLSL3, defines: { DIRS: 2 }, uniforms: u,
                    vertexShader: 'void main(){gl_Position=vec4(position.xy,0.,1.);}',
                    fragmentShader: probeFrag,
                });
                const sc = new T.Scene();
                sc.add(new T.Mesh(new T.PlaneGeometry(2, 2), mat));
                const cam = new T.Camera();
                const rt = new T.WebGLRenderTarget(W, H, {type: T.FloatType, format: T.RGBAFormat, depthBuffer: false});
                const buf = new Float32Array(W * H * 4);

                function run64(tree) {
                    Object.assign(u, tree.uniforms);
                    mat.needsUpdate = true;
                    const sums = new Float64Array(W);
                    for (let f = 0; f < 64; f++) {
                        u.frame.value = f;
                        R.setRenderTarget(rt); R.render(sc, cam);
                        R.readRenderTargetPixels(rt, 0, 0, W, H, buf);
                        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) sums[x] += buf[(y * W + x) * 4] / H;
                    }
                    return Array.from(sums, v => v / 64);
                }

                // Thin-plate suite
                const gaps = [0.02, 0.04, 0.08, 0.12, 0.20];
                const thinPlate = {};
                u.probeMode.value = 0;
                for (const gap of gaps) {
                    u.gap.value = gap;
                    const plate = new T.Mesh(new T.BoxGeometry(0.5, 0.5, 0.02)); plate.position.z = gap + 0.01;
                    const floor = new T.Mesh(new T.BoxGeometry(4, 4, 0.1)); floor.position.z = -0.05;
                    const tree = buildBVH([floor, plate], T, {gridSize: 0, range: u.contactRange.value + u.contactBias.value});
                    thinPlate[gap] = run64(tree);
                    tree.dispose(); plate.geometry.dispose(); floor.geometry.dispose();
                }

                // Curved geometry probe
                u.probeMode.value = 1;
                const cg = new T.BufferGeometry();
                cg.setAttribute('position', new T.BufferAttribute(new Float32Array(curvedTris), 3));
                const cMesh = new T.Mesh(cg);
                const cTree = buildBVH([cMesh], T, {gridSize: 0, range: u.contactRange.value + u.contactBias.value});
                const curved = run64(cTree);
                cTree.dispose(); cg.dispose();
                R.setRenderTarget(null); rt.dispose(); mat.dispose(); sc.children[0].geometry.dispose();
                return { thinPlate, curved };
            }""", {"curvedTris": curved_ref["tris"], "curvedXs": curved_ref["xs"]})

            # Compute exact thin-plate references (16,384 rays)
            n_ref = 16384
            ri = np.arange(n_ref, dtype=np.float64)
            ru = (ri + 0.5) / n_ref
            rphi = ri * 2.399963229728653
            wx, wy, wz = np.sqrt(ru) * np.cos(rphi), np.sqrt(ru) * np.sin(rphi), np.sqrt(1.0 - ru)
            xs_tp = np.linspace(-0.4, 0.4, 129)
            tp_summary = {}
            tp_maes = []
            for gap in [0.02, 0.04, 0.08, 0.12, 0.20]:
                with np.errstate(divide="ignore", invalid="ignore"):
                    a = (-0.25 - xs_tp[:, None]) / wx[None, :]
                    b = (0.25 - xs_tp[:, None]) / wx[None, :]
                    c = -0.25 / wy[None, :]
                    e = 0.25 / wy[None, :]
                    enter = np.maximum(0.0, np.maximum(np.minimum(a, b), np.maximum(np.minimum(c, e), gap / wz[None, :])))
                    exit_ = np.minimum(1.12, np.minimum(np.maximum(a, b), np.minimum(np.maximum(c, e), (gap + 0.02) / wz[None, :])))
                    ref_tp = 1.0 - (exit_ >= enter).mean(axis=1)
                pred_tp = np.array(probes["thinPlate"][str(gap)], dtype=np.float64)
                mae_tp = float(np.mean(np.abs(pred_tp - ref_tp)))
                tp_maes.append(mae_tp)
                tp_summary[str(gap)] = {
                    "center_pred": float(pred_tp[64]),
                    "center_ref": float(ref_tp[64]),
                    "mae": mae_tp,
                }

            curved_pred = np.array(probes["curved"], dtype=np.float64)
            curved_ref_arr = np.array(curved_ref["ref"], dtype=np.float64)
            curved_metrics = {
                "mae": float(np.mean(np.abs(curved_pred - curved_ref_arr))),
                "max_err": float(np.max(np.abs(curved_pred - curved_ref_arr))),
                "under_occ": float(np.mean(np.maximum(0.0, curved_pred - curved_ref_arr))),
            }

            # 6. Real GPU Timings (Wall readback-closed + EXT_disjoint_timer_query_webgl2)
            n_blocks = 4 if quick else 8
            timings = page.evaluate("""async ({eye, target, nBlocks}) => {
                const d = aoDemo, gl = d.renderer.getContext();
                const ext = gl.getExtension('EXT_disjoint_timer_query_webgl2');
                d.setDynamicAO(false);
                d.movers.forEach(m => m.object.visible = false);
                d.setAOScale(0.5);
                d.setMask(true);
                const px = new Uint8Array(4);
                function sync() { gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px); }
                function setOrbit(deg) {
                    const ang = deg * Math.PI / 180;
                    const x = eye[0] - target[0], y = eye[1] - target[1];
                    d.camera.position.set(target[0] + x * Math.cos(ang) - y * Math.sin(ang),
                                          target[1] + x * Math.sin(ang) + y * Math.cos(ang), eye[2]);
                    d.camera.lookAt(...target);
                    d.camera.updateMatrixWorld();
                }
                async function measureBatch(fn, count) {
                    let q = null;
                    if (ext) { q = gl.createQuery(); gl.beginQuery(ext.TIME_ELAPSED_EXT, q); }
                    const t0 = performance.now();
                    for (let i = 0; i < count; i++) fn(i);
                    if (q) gl.endQuery(ext.TIME_ELAPSED_EXT);
                    sync();
                    const wallMs = (performance.now() - t0) / count;
                    let gpuMs = NaN;
                    if (q) {
                        for (let tries = 0; tries < 200; tries++) {
                            if (gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE)) {
                                gpuMs = gl.getQueryParameter(q, gl.QUERY_RESULT) / 1e6 / count;
                                break;
                            }
                            await new Promise(r => setTimeout(r, 1));
                        }
                        gl.deleteQuery(q);
                    }
                    return { wallMs, gpuMs };
                }
                // Warm-up
                setOrbit(0); d.traced.frame.value = 0; d.resetHistory();
                for (let i = 0; i < 16; i++) d.render(64);
                sync();

                const wallEarly = [], wallOrbit = [], wallSettled = [];
                const gpuEarly = [], gpuOrbit = [], gpuSettled = [];
                for (let b = 0; b < nBlocks; b++) {
                    // Early frames 0..15
                    setOrbit(0); d.traced.frame.value = 0; d.resetHistory(); sync();
                    const e = await measureBatch(() => d.render(64), 16);
                    wallEarly.push(e.wallMs); if (Number.isFinite(e.gpuMs)) gpuEarly.push(e.gpuMs);

                    // Settled frames 64..79
                    for (let i = 16; i < 64; i++) d.render(64);
                    sync();
                    const s = await measureBatch(() => d.render(64), 16);
                    wallSettled.push(s.wallMs); if (Number.isFinite(s.gpuMs)) gpuSettled.push(s.gpuMs);

                    // Orbit 1 deg/frame (warmed 16 frames first)
                    for (let i = 0; i < 16; i++) { setOrbit(i); d.render(64); }
                    sync();
                    const o = await measureBatch(i => { setOrbit(16 + i); d.render(64); }, 16);
                    wallOrbit.push(o.wallMs); if (Number.isFinite(o.gpuMs)) gpuOrbit.push(o.gpuMs);
                }
                const med = arr => { const s = [...arr].sort((a,b)=>a-b); return s.length ? s[s.length>>1] : NaN; };
                return {
                    wall_early_ms: med(wallEarly), wall_orbit_ms: med(wallOrbit), wall_settled_ms: med(wallSettled),
                    gpu_early_ms: med(gpuEarly), gpu_orbit_ms: med(gpuOrbit), gpu_settled_ms: med(gpuSettled),
                };
            }""", {"eye": main_ref["eye"], "target": main_ref["target"], "nBlocks": n_blocks})

            browser.close()
    finally:
        httpd.shutdown()

    # Aggregate fitting & held-out summaries
    def pool_metric(views, frame_str, key):
        return float(np.mean([view_scores[v][frame_str][key] for v in views]))

    summary = {
        "tag": tag,
        "elapsed_s": round(time.perf_counter() - t0_all, 1),
        "env": env_info,
        "console_errors": console_errors,
        "page_errors": page_errors,
        "pooled": {
            "fit_mae_1": pool_metric(VIEWS_FIT, "1", "mae"),
            "fit_mae_4": pool_metric(VIEWS_FIT, "4", "mae"),
            "fit_mae_16": pool_metric(VIEWS_FIT, "16", "mae"),
            "fit_mae_64": pool_metric(VIEWS_FIT, "64", "mae"),
            "held_mae_1": pool_metric(VIEWS_HELD, "1", "mae"),
            "held_mae_4": pool_metric(VIEWS_HELD, "4", "mae"),
            "held_mae_16": pool_metric(VIEWS_HELD, "16", "mae"),
            "held_mae_64": pool_metric(VIEWS_HELD, "64", "mae"),
            "contact_mae_16": pool_metric(ALL_VIEWS, "16", "contact_mae"),
            "contact_mae_64": pool_metric(ALL_VIEWS, "64", "contact_mae"),
            "under_occ_16": pool_metric(ALL_VIEWS, "16", "under_occ"),
            "under_occ_64": pool_metric(ALL_VIEWS, "64", "under_occ"),
            "over_occ_64": pool_metric(ALL_VIEWS, "64", "over_occ"),
            "cvar5_16": pool_metric(ALL_VIEWS, "16", "cvar5"),
            "cvar5_64": pool_metric(ALL_VIEWS, "64", "cvar5"),
            "oklab_mae_64": pool_metric(ALL_VIEWS, "64", "oklab_mae"),
            "flip_mean_64": pool_metric(ALL_VIEWS, "64", "flip_mean"),
            "flip_p95_64": pool_metric(ALL_VIEWS, "64", "flip_p95"),
            "gmsd_64": pool_metric(ALL_VIEWS, "64", "gmsd"),
        },
        "temporal": {
            "conv_auc": conv_auc,
            "disocc_mae_1_4": disocc_mae,
            "flicker_static": flicker_static,
            **orbit_metrics,
        },
        "diagnostics": {
            "pix416854_ref1.000": main_diag.get("pix416854"),
            "pix575366_ref0.795": main_diag.get("pix575366"),
        },
        "movers": mover_metrics,
        "thin_plate": {"mean_mae64": float(np.mean(tp_maes)), "gaps": tp_summary},
        "curved_fixture": curved_metrics,
        "timings": timings,
        "main_curve": main_curve,
        "views": view_scores,
    }

    (out_dir / "scorecard.json").write_text(json.dumps(summary, indent=2))

    # Print concise scorecard table
    p_s = summary["pooled"]
    t_s = summary["temporal"]
    tm = summary["timings"]
    print(f"\n=== Judge v2 Scorecard [{tag}] ({summary['elapsed_s']}s on {env_info['renderer']}) ===")
    print(f"GL errors: allZero={all(v == 0 for v in env_info['modeErrors'].values())} | console_err={len(console_errors)} page_err={len(page_errors)}")
    print(f"Fit  MAE (1/4/16/64): {p_s['fit_mae_1']:.5f} / {p_s['fit_mae_4']:.5f} / {p_s['fit_mae_16']:.5f} / {p_s['fit_mae_64']:.5f}")
    print(f"Held MAE (1/4/16/64): {p_s['held_mae_1']:.5f} / {p_s['held_mae_4']:.5f} / {p_s['held_mae_16']:.5f} / {p_s['held_mae_64']:.5f}")
    print(f"Contact@64: MAE={p_s['contact_mae_64']:.5f}  underOcc(leak)={p_s['under_occ_64']:.5f}  overOcc(halo)={p_s['over_occ_64']:.5f}")
    print(f"Perceptual@64: Oklab={p_s['oklab_mae_64']:.5f}  FLIP={p_s['flip_mean_64']:.5f} (p95={p_s['flip_p95_64']:.4f})  GMSD={p_s['gmsd_64']:.5f}  CVaR5={p_s['cvar5_64']:.5f}")
    print(f"Temporal: ConvAUC={t_s['conv_auc']:.5f}  Disocc(1..4)={t_s['disocc_mae_1_4']:.5f}  Orbit3deg={t_s['orbit_3deg_end_mae']:.5f}  Settle4={t_s['orbit_settle4_mae']:.5f}  Flicker={t_s['flicker_static']:.5f}")
    print(f"Diag: pix416854={summary['diagnostics']['pix416854_ref1.000']:.4f} (ref 1.000) | pix575366={summary['diagnostics']['pix575366_ref0.795']:.4f} (ref 0.795)")
    print(f"Movers: Frozen all/near={mover_metrics['frozen']['all_mae']:.5f}/{mover_metrics['frozen']['near_mae']:.5f} | Cont all/near={mover_metrics['continuous']['all_mae']:.5f}/{mover_metrics['continuous']['near_mae']:.5f}")
    print(f"Probes: ThinPlate meanMAE={summary['thin_plate']['mean_mae64']:.5f} (20cm={tp_summary['0.2']['center_pred']:.4f} vs {tp_summary['0.2']['center_ref']:.4f}) | Curved MAE={curved_metrics['mae']:.5f}")
    print(f"Timing (ms/f): Wall early/orbit/settled = {tm['wall_early_ms']:.3f} / {tm['wall_orbit_ms']:.3f} / {tm['wall_settled_ms']:.3f} | GPU = {tm['gpu_early_ms']:.3f} / {tm['gpu_orbit_ms']:.3f} / {tm['gpu_settled_ms']:.3f}")
    print(f"Startup: contactBuild={env_info['contactBuildMs']:.1f}ms ({(env_info['contactBytes'] or 0)/1024:.1f} KiB) | moverBake={env_info['bakeSyncMs']:.1f}ms")
    print(f"Artifacts saved to: {out_dir}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--url", default="/index.html")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run_judge(args.tag, args.url, args.quick)
