#!/usr/bin/env python3
"""Real Apple M1 Chrome WebGL2 test for fast-sdf-ao package exports:
  - loadSDF('./assets/atrium.f16.gz', ...)
  - patchMaterialWithSDFAO(THREE.MeshStandardMaterial, ...)
  - TracedSDFAO.create(...)
Verifies compilation, non-empty rendered output, and 0 WebGL / console / page errors.
"""
import functools
import http.server
import socketserver
import threading
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]

HTML = """<!doctype html>
<html><head><meta charset="utf-8"><link rel="icon" href="data:,">
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js"}}</script>
</head><body>
<script type="module">
import * as THREE from 'three';
import { QUALITY_PRESETS, loadSDF, patchMaterialWithSDFAO, TracedSDFAO } from '/src/index.js';
import {
  ATRIUM_SDF_DIMS, ATRIUM_SDF_MIN, ATRIUM_SDF_SPAN, ATRIUM_CAMERA,
  buildAtriumThreeGeometry,
} from '/assets/atrium-scene.js';

const renderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: true });
renderer.setSize(640, 360, false);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color('#17212e');
scene.add(new THREE.HemisphereLight(0xdce8ff, 0x4a433b, 0.9));
const sun = new THREE.DirectionalLight(0xfff5e4, 1.2);
sun.position.set(-4, -5.5, 10);
scene.add(sun);

const camera = new THREE.PerspectiveCamera(37, 640 / 360, 0.1, 100);
camera.up.set(0, 0, 1);
camera.position.set(...ATRIUM_CAMERA.position);
camera.lookAt(...ATRIUM_CAMERA.target);

const field = await loadSDF('/assets/atrium.f16.gz', { dims: ATRIUM_SDF_DIMS, format: 'f16', THREE });
const { geometry } = buildAtriumThreeGeometry(THREE);
const material = new THREE.MeshStandardMaterial({ color: 0xd8cfc0, roughness: 0.55, metalness: 0.05 });
patchMaterialWithSDFAO(material, {
  field,
  min: ATRIUM_SDF_MIN,
  span: ATRIUM_SDF_SPAN,
  radius: 1.12,
  intensity: 1.0,
  multiBounce: true,
});

const mesh = new THREE.Mesh(geometry, material);
scene.add(mesh);

const gl = renderer.getContext();
const drain = () => {
  const errs = [];
  let e;
  while ((e = gl.getError()) !== gl.NO_ERROR) errs.push(e);
  return errs;
};
drain();

// Render with patchMaterialWithSDFAO on MeshStandardMaterial
renderer.render(scene, camera);
gl.finish();
const patchErrs = drain();

const px = new Uint8Array(640 * 360 * 4);
gl.readPixels(0, 0, 640, 360, gl.RGBA, gl.UNSIGNED_BYTE, px);
let sum = 0;
for (let i = 0; i < px.length; i += 4) sum += px[i] + px[i + 1] + px[i + 2];
const meanLuma = sum / (640 * 360 * 3 * 255);

// Also test TracedSDFAO wrapper + 4-tier quality switching (low -> medium -> high -> ultra -> low)
const traced = await TracedSDFAO.create(renderer, scene, camera, {
  THREE,
  field,
  min: ATRIUM_SDF_MIN,
  span: ATRIUM_SDF_SPAN,
  staticMeshes: [mesh],
  width: 640,
  height: 360,
  bentNormals: true,
  quality: 'high',
});
const out1 = traced.update();
const out2 = traced.update();
const tiersVisited = [];
for (const tier of ['low', 'medium', 'high', 'ultra', 'low']) {
  traced.setQuality(tier);
  tiersVisited.push(traced.quality);
  traced.update();
}
gl.finish();
const tracedErrs = drain();
const hasTextures = Boolean(out1.aoTex && out1.normalTex && out1.bentTex && out2.aoTex);
traced.dispose();

window.__pkgTestResult = {
  patchErrs,
  tracedErrs,
  meanLuma,
  hasTextures,
  tiersVisited,
  presetCount: Object.keys(QUALITY_PRESETS).length,
};
</script>
</body></html>
"""


def main():
    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/__pkg_test.html":
                data = HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            return super().do_GET()

        def log_message(self, format, *args):
            pass

    handler = functools.partial(Handler, directory=str(ROOT))
    socketserver.TCPServer.allow_reuse_address = True
    srv = socketserver.TCPServer(("127.0.0.1", 0), handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    console_errors = []
    page_errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="chrome",
            headless=True,
            args=["--use-gl=angle", "--use-angle=metal", "--enable-webgl", "--ignore-gpu-blocklist"],
        )
        page = browser.new_page(viewport={"width": 700, "height": 420})
        page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: page_errors.append(str(e)))
        page.goto(f"http://127.0.0.1:{port}/__pkg_test.html", wait_until="networkidle")
        page.wait_for_function("() => window.__pkgTestResult !== undefined", timeout=30000)
        res = page.evaluate("() => window.__pkgTestResult")
        browser.close()
    srv.shutdown()

    print("WebGL2 package smoke test:", res)
    print(f"Console errors: {len(console_errors)} | Page errors: {len(page_errors)}")
    if console_errors:
        print("Console errors:", console_errors)
    if page_errors:
        print("Page errors:", page_errors)
    assert len(res["patchErrs"]) == 0, f"patchMaterialWithSDFAO GL errors: {res['patchErrs']}"
    assert len(res["tracedErrs"]) == 0, f"TracedSDFAO GL errors: {res['tracedErrs']}"
    assert res["meanLuma"] > 0.05, f"Rendered frame too dark: {res['meanLuma']}"
    assert res["hasTextures"], "TracedSDFAO did not return textures"
    assert len(console_errors) == 0 and len(page_errors) == 0
    print("WebGL2 package smoke test PASSED (0 GL errors, 0 console errors)!")


if __name__ == "__main__":
    main()
