#!/usr/bin/env node
// Generates the Sunlit Color-Bleeding Gallery mesh, asserts 100% outward-wound
// triangles (flipped === 0), bakes its exact isotropic FP16 SDF with brute-force
// validation, and writes assets/gallery.f16.gz.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import zlib from 'node:zlib';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import {
  GALLERY_SDF_MIN,
  GALLERY_SDF_SPAN,
  GALLERY_SDF_DIMS,
  buildGalleryGeometryData,
} from '../assets/gallery-scene.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..');

const args = process.argv.slice(2);
const getArg = (flag) => {
  const i = args.indexOf(flag);
  return i >= 0 ? args[i + 1] : null;
};
const exportTrisPath = getArg('--export-tris');
const exportGiTrisPath = getArg('--export-gi-tris');
const skipBake = args.includes('--tris-only');

const data = buildGalleryGeometryData();
let minX = Infinity, minY = Infinity, minZ = Infinity;
let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
for (let i = 0; i < data.positions.length; i += 3) {
  const x = data.positions[i], y = data.positions[i + 1], z = data.positions[i + 2];
  if (x < minX) minX = x; if (x > maxX) maxX = x;
  if (y < minY) minY = y; if (y > maxY) maxY = y;
  if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
}

// Strict outward-winding assertion across all triangles
let flipped = 0;
let degenerate = 0;
for (let t = 0; t < data.triangleCount; t++) {
  const i0 = data.indices[t * 3] * 3;
  const i1 = data.indices[t * 3 + 1] * 3;
  const i2 = data.indices[t * 3 + 2] * 3;
  const ax = data.positions[i0], ay = data.positions[i0 + 1], az = data.positions[i0 + 2];
  const bx = data.positions[i1], by = data.positions[i1 + 1], bz = data.positions[i1 + 2];
  const cx = data.positions[i2], cy = data.positions[i2 + 1], cz = data.positions[i2 + 2];
  const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
  const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
  const fnx = e1y * e2z - e1z * e2y;
  const fny = e1z * e2x - e1x * e2z;
  const fnz = e1x * e2y - e1y * e2x;
  const flen = Math.hypot(fnx, fny, fnz);
  if (flen < 1e-10) {
    degenerate++;
    continue;
  }
  const vnx = data.normals[i0] + data.normals[i1] + data.normals[i2];
  const vny = data.normals[i0 + 1] + data.normals[i1 + 1] + data.normals[i2 + 1];
  const vnz = data.normals[i0 + 2] + data.normals[i1 + 2] + data.normals[i2 + 2];
  if (fnx * vnx + fny * vny + fnz * vnz <= 0) {
    flipped++;
  }
}

console.log(
  `Gallery mesh: ${data.vertexCount.toLocaleString()} vertices, ${data.triangleCount.toLocaleString()} triangles (flipped=${flipped}, degenerate=${degenerate})`
);
console.log(
  `Bounds: X [${minX.toFixed(3)}, ${maxX.toFixed(3)}], Y [${minY.toFixed(3)}, ${maxY.toFixed(3)}], Z [${minZ.toFixed(3)}, ${maxZ.toFixed(3)}]`
);
if (flipped !== 0) {
  console.error(`ERROR: Gallery mesh has ${flipped} inward-wound triangles!`);
  process.exit(1);
}

if (exportTrisPath) {
  const tris = new Float32Array(data.triangleCount * 9);
  for (let t = 0; t < data.triangleCount; t++) {
    const i0 = data.indices[t * 3] * 3;
    const i1 = data.indices[t * 3 + 1] * 3;
    const i2 = data.indices[t * 3 + 2] * 3;
    const o = t * 9;
    tris[o]     = data.positions[i0];     tris[o + 1] = data.positions[i0 + 1]; tris[o + 2] = data.positions[i0 + 2];
    tris[o + 3] = data.positions[i1];     tris[o + 4] = data.positions[i1 + 1]; tris[o + 5] = data.positions[i1 + 2];
    tris[o + 6] = data.positions[i2];     tris[o + 7] = data.positions[i2 + 1]; tris[o + 8] = data.positions[i2 + 2];
  }
  fs.mkdirSync(path.dirname(path.resolve(exportTrisPath)), { recursive: true });
  fs.writeFileSync(exportTrisPath, Buffer.from(tris.buffer));
  console.log(`Wrote ${exportTrisPath} (${(tris.byteLength / 1024).toFixed(1)} KiB)`);
}

if (exportGiTrisPath) {
  // 15 floats per triangle: v0(3), v1(3), v2(3), unitNormal(3), avgAlbedo(3)
  const giTris = new Float32Array(data.triangleCount * 15);
  for (let t = 0; t < data.triangleCount; t++) {
    const i0 = data.indices[t * 3] * 3;
    const i1 = data.indices[t * 3 + 1] * 3;
    const i2 = data.indices[t * 3 + 2] * 3;
    const o = t * 15;
    const ax = data.positions[i0], ay = data.positions[i0 + 1], az = data.positions[i0 + 2];
    const bx = data.positions[i1], by = data.positions[i1 + 1], bz = data.positions[i1 + 2];
    const cx = data.positions[i2], cy = data.positions[i2 + 1], cz = data.positions[i2 + 2];
    giTris[o]     = ax; giTris[o + 1] = ay; giTris[o + 2] = az;
    giTris[o + 3] = bx; giTris[o + 4] = by; giTris[o + 5] = bz;
    giTris[o + 6] = cx; giTris[o + 7] = cy; giTris[o + 8] = cz;
    const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
    const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
    let nx = e1y * e2z - e1z * e2y;
    let ny = e1z * e2x - e1x * e2z;
    let nz = e1x * e2y - e1y * e2x;
    const nl = Math.hypot(nx, ny, nz) || 1;
    giTris[o + 9] = nx / nl; giTris[o + 10] = ny / nl; giTris[o + 11] = nz / nl;
    giTris[o + 12] = (data.colors[i0]     + data.colors[i1]     + data.colors[i2])     / 3;
    giTris[o + 13] = (data.colors[i0 + 1] + data.colors[i1 + 1] + data.colors[i2 + 1]) / 3;
    giTris[o + 14] = (data.colors[i0 + 2] + data.colors[i1 + 2] + data.colors[i2 + 2]) / 3;
  }
  fs.mkdirSync(path.dirname(path.resolve(exportGiTrisPath)), { recursive: true });
  fs.writeFileSync(exportGiTrisPath, Buffer.from(giTris.buffer));
  console.log(`Wrote ${exportGiTrisPath} (${(giTris.byteLength / 1024).toFixed(1)} KiB)`);
}

if (skipBake) process.exit(0);

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'fast-sdf-ao-gallery-'));
const objPath = path.join(tmpDir, 'gallery.obj');
const f16Path = path.join(tmpDir, 'gallery.f16');
const bakerBin = path.join(os.tmpdir(), 'fast-sdf-ao-bake');
const bakerSrc = path.join(repoRoot, 'tools', 'bake_sdf.cpp');

const lines = ['# Fast-SDF-AO Sunlit Color-Bleeding Gallery procedural mesh'];
for (let i = 0; i < data.positions.length; i += 3) {
  lines.push(`v ${data.positions[i]} ${data.positions[i + 1]} ${data.positions[i + 2]}`);
}
for (let i = 0; i < data.indices.length; i += 3) {
  lines.push(`f ${data.indices[i] + 1} ${data.indices[i + 1] + 1} ${data.indices[i + 2] + 1}`);
}
fs.writeFileSync(objPath, lines.join('\n') + '\n');

const needCompile = !fs.existsSync(bakerBin) || fs.statSync(bakerBin).mtimeMs < fs.statSync(bakerSrc).mtimeMs;
if (needCompile) {
  console.log('Compiling tools/bake_sdf.cpp...');
  const cxxFlags = ['-O3', '-std=c++17', '-pthread'];
  if (process.platform === 'darwin') {
    try {
      const sdk = execFileSync('xcrun', ['--show-sdk-path'], { encoding: 'utf8' }).trim();
      const cxxInc = path.join(sdk, 'usr', 'include', 'c++', 'v1');
      if (fs.existsSync(cxxInc)) cxxFlags.push('-isystem', cxxInc);
    } catch {}
  }
  execFileSync('c++', [...cxxFlags, bakerSrc, '-o', bakerBin], { stdio: 'inherit' });
}

const bakeArgs = [
  '--obj', objPath,
  '--lo', ...GALLERY_SDF_MIN.map(String),
  '--span', ...GALLERY_SDF_SPAN.map(String),
  '--res-xyz', ...GALLERY_SDF_DIMS.map(String),
  '--clip', '1.5',
  '--reach', '1.12',
  '--f16',
  '--validate', '2000',
  '--out', f16Path,
];
console.log('Baking Gallery SDF:', bakeArgs.join(' '));
execFileSync(bakerBin, bakeArgs, { stdio: 'inherit' });

const raw = fs.readFileSync(f16Path);
const gz = zlib.gzipSync(raw, { level: 9 });
const outGz = path.join(repoRoot, 'assets', 'gallery.f16.gz');
fs.writeFileSync(outGz, gz);
console.log(`Wrote ${path.relative(repoRoot, outGz)}: ${(raw.byteLength / 1024).toFixed(1)} KiB raw -> ${(gz.byteLength / 1024).toFixed(1)} KiB gz`);

fs.rmSync(tmpDir, { recursive: true, force: true });
