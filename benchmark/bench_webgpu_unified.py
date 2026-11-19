#!/usr/bin/env python3
"""Real Apple M1 Metal-3 WebGPU benchmark for the Unified WebGPU Compute Backend (webgpu.html + src/webgpu.js)."""
import http.server
import json
import socketserver
import threading
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "nextgen"
OUT.mkdir(parents=True, exist_ok=True)


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, *args):
        pass


def main():
    httpd = socketserver.TCPServer(("127.0.0.1", 0), QuietHandler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    console_errs, page_errs = [], []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                channel="chrome",
                headless=True,
                args=[
                    "--enable-unsafe-webgpu",
                    "--enable-features=Vulkan,WebGPU",
                    "--use-angle=metal",
                    "--ignore-gpu-blocklist",
                    "--enable-dawn-features=allow_unsafe_apis",
                ],
            )
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            page.on("console", lambda m: console_errs.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e: page_errs.append(str(e)))
            page.goto(f"http://127.0.0.1:{port}/webgpu.html", wait_until="networkidle")
            page.wait_for_function("() => window.wgpuDemo !== undefined", timeout=25000)

            results = page.evaluate("""async () => {
              const d = wgpuDemo;
              d.stopLoop();
              const dev = d.gpu.device;
              const waitGPU = () => dev.queue.onSubmittedWorkDone();
              const benchScene = async (name) => {
                await d.selectScene(name);
                await waitGPU();
                // Cold frames (0..15)
                const t0 = performance.now();
                for (let f = 0; f < 16; f++) d.renderOnce();
                await waitGPU();
                const coldMs = (performance.now() - t0) / 16;

                // Converge to settled (frames 16..80)
                for (let f = 0; f < 64; f++) d.renderOnce();
                await waitGPU();
                const t1 = performance.now();
                for (let f = 0; f < 24; f++) d.renderOnce();
                await waitGPU();
                const settledMs = (performance.now() - t1) / 24;
                const settledGpuMs = d.gpu.lastGpuMs;
                const settledSkip = d.gpu.lastSkipPct;

                // Orbit 3 deg/frame
                const eye0 = d.camera.position.clone();
                const tgt = d.controls.target.clone();
                const off = eye0.clone().sub(tgt);
                const t2 = performance.now();
                for (let f = 0; f < 16; f++) {
                  const a = (f + 1) * 3 * Math.PI / 180;
                  const ca = Math.cos(a), sa = Math.sin(a);
                  d.camera.position.set(tgt.x + off.x * ca - off.y * sa, tgt.y + off.x * sa + off.y * ca, eye0.z);
                  d.renderOnce();
                }
                await waitGPU();
                const orbitMs = (performance.now() - t2) / 16;

                // Camera-cut revisit to original view (tests instant SHaRC world-space cache reuse!)
                d.camera.position.copy(eye0);
                d.renderOnce();
                await waitGPU();
                await new Promise(r => setTimeout(r, 20));
                const revisitSkip = d.gpu.lastSkipPct;

                return {
                  coldMs: +coldMs.toFixed(3),
                  settledMs: +settledMs.toFixed(3),
                  settledGpuMs: +settledGpuMs.toFixed(3),
                  orbitMs: +orbitMs.toFixed(3),
                  settledSkipPct: +settledSkip.toFixed(1),
                  revisitSkipPct: +revisitSkip.toFixed(1),
                };
              };

              const out = {
                adapter: d.gpu.adapterInfo,
                hasTimestamp: d.gpu.hasTimestamp,
                scenes: {},
                tiers: {},
              };

              // Benchmark all 3 scenes on High
              d.gpu.setQuality('high');
              for (const sc of ['gallery', 'atrium', 'archive']) {
                out.scenes[sc] = await benchScene(sc);
              }

              // Benchmark all 4 quality tiers on Gallery
              await d.selectScene('gallery');
              for (const tier of ['low', 'medium', 'high', 'ultra']) {
                d.gpu.setQuality(tier);
                d.gpu.clearHash();
                for (let f = 0; f < 80; f++) d.renderOnce();
                await waitGPU();
                const ts = performance.now();
                for (let f = 0; f < 24; f++) d.renderOnce();
                await waitGPU();
                out.tiers[tier] = {
                  settledMs: +((performance.now() - ts) / 24).toFixed(3),
                  gpuMs: +d.gpu.lastGpuMs.toFixed(3),
                  skipPct: +d.gpu.lastSkipPct.toFixed(1),
                };
              }
              return out;
            }""")

            # Capture visual screenshots
            page.evaluate("""async () => {
              const d = wgpuDemo;
              d.gpu.setQuality('high');
              await d.selectScene('gallery');
              d.gpu.viewMode = 0;
              d.setEmissive(false);
              for (let f = 0; f < 72; f++) d.renderOnce();
              await d.gpu.device.queue.onSubmittedWorkDone();
            }""")
            page.locator("canvas").screenshot(path=str(OUT / "webgpu_gallery_gi.png"))

            page.evaluate("""async () => {
              const d = wgpuDemo;
              d.setEmissive(true, 0.7);
              for (let f = 0; f < 24; f++) d.renderOnce();
              await d.gpu.device.queue.onSubmittedWorkDone();
            }""")
            page.locator("canvas").screenshot(path=str(OUT / "webgpu_gallery_orb.png"))

            page.evaluate("""async () => {
              const d = wgpuDemo;
              d.setEmissive(false);
              await d.selectScene('archive');
              d.gpu.viewMode = 1;
              for (let f = 0; f < 72; f++) d.renderOnce();
              await d.gpu.device.queue.onSubmittedWorkDone();
            }""")
            page.locator("canvas").screenshot(path=str(OUT / "webgpu_archive_ao.png"))

            browser.close()
    finally:
        httpd.shutdown()

    results["console_errors"] = console_errs
    results["page_errors"] = page_errs
    (OUT / "webgpu_unified_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
