#!/usr/bin/env python3
"""Run the WebGPU Compute World-Space Spatial Hash AO Cache benchmark on Apple M1 Metal-3."""
import functools
import http.server
import json
import socketserver
import threading
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
REF_MAIN = ROOT / "output/playwright/traced-run2/main.json"
OUT_PATH = ROOT / "output/nextgen/webgpu_hash_ao.json"


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main():
    httpd = socketserver.TCPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(ROOT)))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    ref_main = json.loads(REF_MAIN.read_text())

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                channel="chrome",
                headless=True,
                args=["--use-gl=angle", "--use-angle=metal", "--ignore-gpu-blocklist", "--enable-unsafe-webgpu"],
            )
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.goto(f"http://127.0.0.1:{port}/benchmark/webgpu-hash-ao.html")
            page.wait_for_function("() => window.webgpuHashDemo !== undefined", timeout=30000)
            res = page.evaluate("async (ref) => await window.webgpuHashDemo.runBenchmark(ref)", ref_main)
            info = page.evaluate("() => window.webgpuHashDemo.adapterInfo")
            browser.close()
    finally:
        httpd.shutdown()

    out = {"adapter": info, "errors": errors, "result": res}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    c = res["cold"]
    m = res["ms"]
    print(f"WebGPU Adapter: {info['vendor']} / {info['arch']} | errors={len(errors)}")
    print(f"Cold MAE (1/4/16/64): {c['f1']:.5f} / {c['f4']:.5f} / {c['f16']:.5f} / {c['f64']:.5f}")
    print(f"Camera-Cut Revisit Frame-1 MAE: {res['revisit_frame1_mae']:.5f} (skipRate={res['revisit_skip_rate']*100:.1f}%)")
    print(f"Settled Skip Rate: {res['settled_skip_rate']*100:.1f}% | Table: {res['table_bytes']/1024/1024:.1f} MiB")
    print(f"GPU Time (ms/f): cold={m['cold_early_ms']:.3f} | orbit3deg={m['orbit_3deg_ms']:.3f} | settled={m['settled_ms']:.3f} | direct_uncached={m['direct_uncached_ms']:.3f}")


if __name__ == "__main__":
    main()
