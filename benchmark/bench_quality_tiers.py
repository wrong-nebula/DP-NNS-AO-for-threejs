#!/usr/bin/env python3
"""Benchmark & verify all 4 Quality Presets (low, medium, high, ultra) on Apple M1 Chrome WebGL2.

Checks:
  1. Zero WebGL errors, zero console errors, zero page errors across all 3 scenes,
     all 4 quality tiers, cycling low -> medium -> high -> ultra -> low, and all 7 AO modes.
  2. Zero texture leaks on quality tier cycling (renderer.info.memory.textures stays bounded).
  3. Wall-clock ms/f (early / moving-sun / settled / orb) across Archive, Atrium, and Gallery.
  4. Saves Gallery visual captures for low / medium / high / ultra / orb to output/nextgen/.
"""
import functools
import http.server
import json
import socketserver
import threading
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "nextgen"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TIERS = ["low", "medium", "high", "ultra"]


def main():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    socketserver.TCPServer.allow_reuse_address = True
    srv = socketserver.TCPServer(("127.0.0.1", 0), handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    console_errors = []
    page_errors = []
    results = {"scenes": {}, "cycleCheck": {}, "errors": {}}

    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="chrome",
            headless=True,
            args=["--use-gl=angle", "--use-angle=metal", "--enable-webgl", "--ignore-gpu-blocklist"],
        )
        page = browser.new_page(viewport={"width": 1340, "height": 900})
        page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: page_errors.append(str(e)))
        page.goto(f"http://127.0.0.1:{port}/index.html", wait_until="networkidle")
        page.wait_for_function("() => window.aoDemo !== undefined", timeout=30000)

        # Pause RAF loop so measurements are deterministic
        page.evaluate("""() => {
          const d = window.aoDemo;
          d.renderer.setAnimationLoop(null);
          document.querySelector('#animate').checked = false;
          d.updateMovers(0);
        }""")

        # 1. Cycle test + leak check across all 3 scenes and all 4 tiers + 7 AO modes
        cycle = page.evaluate("""async () => {
          const d = window.aoDemo;
          const gl = d.renderer.getContext();
          const px = new Uint8Array(4);
          const sync = () => gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
          const drain = () => { const e = []; let c; while ((c = gl.getError()) !== gl.NO_ERROR) e.push(c); return e; };
          drain();
          let glErrors = 0;
          const texCounts = {};
          for (const sc of ['archive', 'atrium', 'gallery']) {
            await d.setScene(sc);
            d.select(64);
            // Cycle 1: warms all lazy targets (e.g. hiz1, GI targets)
            for (const tier of ['low', 'medium', 'high', 'ultra', 'high']) {
              d.setQuality(tier);
              if (d.quality !== tier) throw Error('Quality mismatch: ' + d.quality + ' vs ' + tier);
              d.render(64); sync();
              glErrors += drain().length;
            }
            const afterCycle1 = d.renderer.info.memory.textures;
            // Cycle 2: must allocate 0 net new textures
            for (const tier of ['low', 'medium', 'high', 'ultra', 'low', 'high']) {
              d.setQuality(tier);
              d.render(64); d.render(64); sync();
              glErrors += drain().length;
            }
            const afterCycle2 = d.renderer.info.memory.textures;
            texCounts[sc] = { afterCycle1, afterCycle2 };
          }
          // Verify all 7 AO modes on Archive under 'high'
          await d.setScene('archive');
          d.setQuality('high');
          for (const mode of [0, 4, 14, 34, 64, 32, 36]) {
            d.select(mode);
            d.render(mode); sync();
            glErrors += drain().length;
          }
          d.select(64);
          return { glErrors, texCounts };
        }""")
        results["cycleCheck"] = cycle

        # 2. Benchmark Archive & Atrium across all 4 tiers
        for sc in ["archive", "atrium"]:
            results["scenes"][sc] = {}
            page.evaluate(f"async () => {{ await window.aoDemo.setScene('{sc}'); window.aoDemo.select(64); }}")
            for tier in TIERS:
                m = page.evaluate("""(tier) => {
                  const d = window.aoDemo;
                  const gl = d.renderer.getContext();
                  const px = new Uint8Array(4);
                  const sync = () => gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
                  const med = arr => { const s = arr.slice().sort((a,b) => a - b); return s[s.length >> 1]; };
                  d.setQuality(tier);
                  // Warmup
                  d.resetHistory();
                  for (let i = 0; i < 4; i++) { d.render(64); sync(); }
                  // Measure early (frames 1..10)
                  d.resetHistory();
                  const early = [];
                  for (let i = 0; i < 10; i++) {
                    const t0 = performance.now();
                    d.render(64); sync();
                    early.push(performance.now() - t0);
                  }
                  // Settle
                  for (let i = 0; i < 70; i++) d.render(64);
                  sync();
                  // Measure settled (12 frames)
                  const settled = [];
                  for (let i = 0; i < 12; i++) {
                    const t0 = performance.now();
                    d.render(64); sync();
                    settled.push(performance.now() - t0);
                  }
                  let glErrs = 0; while (gl.getError() !== gl.NO_ERROR) glErrs++;
                  return { earlyMs: +med(early).toFixed(2), settledMs: +med(settled).toFixed(2), glErrs };
                }""", tier)
                results["scenes"][sc][tier] = m

        # 3. Benchmark Gallery (GI + Multi-bounce + Moving Sun + Orb) across all 4 tiers + capture PNGs
        results["scenes"]["gallery"] = {}
        page.evaluate("async () => { await window.aoDemo.setScene('gallery'); window.aoDemo.select(64); }")
        canvas = page.locator("canvas")
        for tier in TIERS:
            m = page.evaluate("""(tier) => {
              const d = window.aoDemo;
              const gl = d.renderer.getContext();
              const px = new Uint8Array(4);
              const sync = () => gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
              const med = arr => { const s = arr.slice().sort((a,b) => a - b); return s[s.length >> 1]; };
              d.setQuality(tier);
              d.setGI(true);
              d.setMultiBounceGI(true);
              d.setEmissive(false);
              d.setSunAngle(22);
              // Warmup
              d.resetHistory();
              for (let i = 0; i < 4; i++) { d.render(64); sync(); }
              // Early (frames 1..10)
              d.resetHistory();
              const early = [];
              for (let i = 0; i < 10; i++) {
                const t0 = performance.now();
                d.render(64); sync();
                early.push(performance.now() - t0);
              }
              // Moving sun (10 frames)
              const movingSun = [];
              for (let i = 0; i < 10; i++) {
                const t0 = performance.now();
                d.setSunAngle(12 + i * 2);
                d.render(64); sync();
                movingSun.push(performance.now() - t0);
              }
              // Reset sun to 22 and settle (140 frames)
              d.setSunAngle(22);
              d.resetHistory();
              for (let i = 0; i < 140; i++) d.render(64);
              sync();
              // Settled (12 frames)
              const settled = [];
              for (let i = 0; i < 12; i++) {
                const t0 = performance.now();
                d.render(64); sync();
                settled.push(performance.now() - t0);
              }
              let glErrs = 0; while (gl.getError() !== gl.NO_ERROR) glErrs++;
              return { earlyMs: +med(early).toFixed(2), movingSunMs: +med(movingSun).toFixed(2), settledMs: +med(settled).toFixed(2), glErrs };
            }""", tier)
            # Capture settled Gallery screenshot for this tier
            canvas.screenshot(path=str(OUT_DIR / f"gallery_tier_{tier}.png"))

            # Also measure moving emissive orb on this tier
            orb_ms = page.evaluate("""() => {
              const d = window.aoDemo;
              const gl = d.renderer.getContext();
              const px = new Uint8Array(4);
              const sync = () => gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
              const med = arr => { const s = arr.slice().sort((a,b) => a - b); return s[s.length >> 1]; };
              d.setEmissive(true);
              d.setEmissiveTime(0.0);
              d.render(64); sync();
              const samples = [];
              for (let i = 1; i <= 10; i++) {
                const t0 = performance.now();
                d.setEmissiveTime(i * 0.25);
                d.render(64); sync();
                samples.push(performance.now() - t0);
              }
              return +med(samples).toFixed(2);
            }""")
            m["orbMs"] = orb_ms
            if tier == "high":
                canvas.screenshot(path=str(OUT_DIR / "gallery_tier_high_orb.png"))
            page.evaluate("() => window.aoDemo.setEmissive(false)")
            results["scenes"]["gallery"][tier] = m

        browser.close()
    srv.shutdown()

    results["errors"] = {
        "cycleGlErrors": cycle["glErrors"],
        "consoleErrors": len(console_errors),
        "pageErrors": len(page_errors),
    }

    out_json = OUT_DIR / "quality_tiers_results.json"
    out_json.write_text(json.dumps(results, indent=2))

    print("\n=== 4-Tier Quality Settings Benchmark (Apple M1 Chrome WebGL2, 1280x720) ===")
    print(f"Cycle check: GL errors={cycle['glErrors']} | texCounts={cycle['texCounts']}")
    print(f"Console errors: {len(console_errors)} | Page errors: {len(page_errors)}")
    print("\n| Tier | Archive (early / settled) | Atrium (early / settled) | Gallery GI (early / sun / settled / orb) |")
    print("|---|---:|---:|---:|")
    for tier in TIERS:
        ar = results["scenes"]["archive"][tier]
        at = results["scenes"]["atrium"][tier]
        ga = results["scenes"]["gallery"][tier]
        print(
            f"| **{tier}** | {ar['earlyMs']:.2f} / {ar['settledMs']:.2f} ms | "
            f"{at['earlyMs']:.2f} / {at['settledMs']:.2f} ms | "
            f"{ga['earlyMs']:.2f} / {ga['movingSunMs']:.2f} / {ga['settledMs']:.2f} / {ga['orbMs']:.2f} ms |"
        )
    print(f"\nSaved JSON: {out_json}")
    assert cycle["glErrors"] == 0, f"GL errors during cycle: {cycle}"
    assert len(console_errors) == 0, f"Console errors: {console_errors}"
    assert len(page_errors) == 0, f"Page errors: {page_errors}"
    for sc, tc in cycle["texCounts"].items():
        assert tc["afterCycle2"] == tc["afterCycle1"], f"Texture leak in {sc}: {tc}"


if __name__ == "__main__":
    main()
