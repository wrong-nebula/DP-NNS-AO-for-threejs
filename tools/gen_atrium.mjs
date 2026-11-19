#!/usr/bin/env node
// Generates the Atrium curved-mesh OBJ, bakes its exact isotropic FP16 SDF
// (160 x 160 x 80, 8.125 cm voxels) with brute-force validation, and writes
// assets/atrium.f16.gz.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import zlib from 'node:zlib';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import {
  ATRIUM_SDF_MIN,
  ATRIUM_SDF_SPAN,
  ATRIUM_SDF_DIMS,
  buildAtriumGeometryData,
} from '../assets/atrium-scene.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..');

const args = process.argv.slice(2);
const exportTrisPath = (() => {
  const i = args.indexOf('--export-tris');
  return i >= 0 ? args[i + 1] : null;
})();
const skipBake = args.includes('--tris-only');

const data = buildAtriumGeometryData();
let minX = Infinity, minY = Infinity, minZ = Infinity;
let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
for (let i = 0; i < data.positions.length; i += 3) {
  const x = data.positions[i], y = data.positions[i + 1], z = data.positions[i + 2];
  if (x < minX) minX = x; if (x > maxX) maxX = x;
  if (y < minY) minY = y; if (y > maxY) maxY = y;
  if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
}
console.log(
  `Atrium mesh: ${data.vertexCount.toLocaleString()} vertices, ${data.triangleCount.toLocaleString()} triangles`
);
console.log(
  `Bounds: X [${minX.toFixed(3)}, ${maxX.toFixed(3)}], Y [${minY.toFixed(3)}, ${maxY.toFixed(3)}], Z [${minZ.toFixed(3)}, ${maxZ.toFixed(3)}]`
);

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

if (skipBake) process.exit(0);

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'fast-sdf-ao-atrium-'));
const objPath = path.join(tmpDir, 'atrium.obj');
const f16Path = path.join(tmpDir, 'atrium.f16');
const bakerBin = path.join(os.tmpdir(), 'fast-sdf-ao-bake');
const bakerSrc = path.join(repoRoot, 'tools', 'bake_sdf.cpp');

// Write OBJ
const lines = ['# Fast-SDF-AO Atrium procedural curved mesh'];
for (let i = 0; i < data.positions.length; i += 3) {
  lines.push(`v ${data.positions[i]} ${data.positions[i + 1]} ${data.positions[i + 2]}`);
}
for (let i = 0; i < data.indices.length; i += 3) {
  lines.push(`f ${data.indices[i] + 1} ${data.indices[i + 1] + 1} ${data.indices[i + 2] + 1}`);
}
fs.writeFileSync(objPath, lines.join('\n') + '\n');

// Compile baker if missing or older than source
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

// Bake exact isotropic FP16 SDF with brute-force validation
const bakeArgs = [
  '--obj', objPath,
  '--lo', ...ATRIUM_SDF_MIN.map(String),
  '--span', ...ATRIUM_SDF_SPAN.map(String),
  '--res-xyz', ...ATRIUM_SDF_DIMS.map(String),
  '--clip', '1.5',
  '--reach', '1.12',
  '--f16',
  '--validate', '2000',
  '--out', f16Path,
];
console.log('Baking Atrium SDF:', bakeArgs.join(' '));
execFileSync(bakerBin, bakeArgs, { stdio: 'inherit' });

// Gzip to assets/atrium.f16.gz
const raw = fs.readFileSync(f16Path);
const gz = zlib.gzipSync(raw, { level: 9 });
const outGz = path.join(repoRoot, 'assets', 'atrium.f16.gz');
fs.writeFileSync(outGz, gz);
console.log(`Wrote ${path.relative(repoRoot, outGz)}: ${(raw.byteLength / 1024).toFixed(1)} KiB raw -> ${(gz.byteLength / 1024).toFixed(1)} KiB gz`);

fs.rmSync(tmpDir, { recursive: true, force: true });
