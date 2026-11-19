#!/usr/bin/env node
// Automated verification for the fast-sdf-ao package:
//   1. package.json metadata & exports
//   2. src/index.js exports & GENERIC_SDF_AO_GLSL structure
//   3. src/tsl.js syntax & exported node factories
//   4. loadSDF round-trip on assets/atrium.f16.gz
//   5. buildContactBVH on non-box curved Atrium geometry
//   6. bin/bake-sdf.mjs --help + small OBJ bake with --validate 200

import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import {
  loadSDF,
  GENERIC_SDF_AO_GLSL,
  patchMaterialWithSDFAO,
  TracedSDFAO,
  buildContactBVH,
  CONTACT_GLSL,
} from '../src/index.js';
import {
  ATRIUM_SDF_DIMS,
  ATRIUM_SDF_MIN,
  ATRIUM_SDF_SPAN,
  buildAtriumGeometryData,
} from '../assets/atrium-scene.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..');

// Minimal Three.js mock for headless Node verification of Data3DTexture & buildContactBVH
class MockTexture {
  constructor(data, width, height, depth) {
    this.image = { data, width, height, depth };
    this.data = data;
    this.needsUpdate = false;
  }
  dispose() {}
}
class MockVector3 {
  constructor(x = 0, y = 0, z = 0) { this.x = x; this.y = y; this.z = z; }
  set(x, y, z) { this.x = x; this.y = y; this.z = z; return this; }
}
class MockMatrix4 {
  constructor() { this.elements = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]; }
  copy(m) { this.elements = m.elements.slice(); return this; }
  multiplyMatrices(a, b) { this.elements = a.elements.slice(); return this; }
}
const MockTHREE = {
  DataTexture: class extends MockTexture { constructor(d, w, h) { super(d, w, h, 1); } },
  Data3DTexture: class extends MockTexture { constructor(d, w, h, z) { super(d, w, h, z); } },
  Vector3: MockVector3,
  Matrix4: MockMatrix4,
  RedFormat: 1028,
  RGBAFormat: 1023,
  RGBAIntegerFormat: 1033,
  RGIntegerFormat: 1031,
  RedIntegerFormat: 1030,
  HalfFloatType: 1016,
  FloatType: 1015,
  UnsignedIntType: 1014,
  LinearFilter: 1006,
  NearestFilter: 1003,
};

console.log('1. Checking package.json & LICENSE...');
const pkg = JSON.parse(fs.readFileSync(path.join(repoRoot, 'package.json'), 'utf8'));
assert.equal(pkg.name, 'fast-sdf-ao');
assert.equal(pkg.type, 'module');
assert.equal(pkg.license, 'MIT');
assert.equal(pkg.exports['.'], './src/index.js');
assert.equal(pkg.exports['./contact-bvh'], './contact-bvh.js');
assert.equal(pkg.exports['./tsl'], './src/tsl.js');
assert.equal(pkg.bin['bake-sdf'], './bin/bake-sdf.mjs');
assert.ok(fs.existsSync(path.join(repoRoot, 'LICENSE')));
console.log('   OK');

console.log('2. Checking src/index.js exports & GLSL...');
assert.equal(typeof loadSDF, 'function');
assert.equal(typeof patchMaterialWithSDFAO, 'function');
assert.equal(typeof TracedSDFAO, 'function');
assert.equal(typeof buildContactBVH, 'function');
assert.equal(typeof CONTACT_GLSL, 'string');
assert.equal(typeof GENERIC_SDF_AO_GLSL, 'string');
assert.ok(GENERIC_SDF_AO_GLSL.includes('sdfAOFourTap'));
assert.ok(GENERIC_SDF_AO_GLSL.includes('multiBounceAO'));
assert.ok(GENERIC_SDF_AO_GLSL.includes('specularOcclusion'));
assert.ok(!GENERIC_SDF_AO_GLSL.includes('wallFace'), 'Generic GLSL must not depend on wallFace');
console.log('   OK');

console.log('3. Checking src/tsl.js syntax & exports...');
const tslSrc = fs.readFileSync(path.join(repoRoot, 'src/tsl.js'), 'utf8');
execFileSync('node', ['--check', path.join(repoRoot, 'src/tsl.js')], { stdio: 'inherit' });
assert.ok(tslSrc.includes("from 'three/tsl'"));
assert.ok(tslSrc.includes('export function createSDFAONode'));
assert.ok(tslSrc.includes('export const createMultiBounceAONode'));
console.log('   OK');

console.log('4. Testing loadSDF round-trip with assets/atrium.f16.gz...');
const atriumGz = fs.readFileSync(path.join(repoRoot, 'assets/atrium.f16.gz'));
const tex = await loadSDF(atriumGz, { dims: ATRIUM_SDF_DIMS, format: 'f16', THREE: MockTHREE });
const expectedVoxels = ATRIUM_SDF_DIMS[0] * ATRIUM_SDF_DIMS[1] * ATRIUM_SDF_DIMS[2];
assert.equal(tex.data.length, expectedVoxels);
assert.equal(tex.format, MockTHREE.RedFormat);
assert.equal(tex.type, MockTHREE.HalfFloatType);
assert.equal(tex.minFilter, MockTHREE.LinearFilter);
assert.equal(tex.needsUpdate, true);
console.log(`   OK (${expectedVoxels.toLocaleString()} FP16 voxels decoded from ${(atriumGz.byteLength / 1024).toFixed(1)} KiB gz)`);

console.log('5. Testing buildContactBVH on non-box curved Atrium geometry...');
const atriumData = buildAtriumGeometryData();
const mockAtriumMesh = {
  isInstancedMesh: false,
  matrixWorld: new MockMatrix4(),
  updateWorldMatrix() {},
  geometry: {
    index: {
      count: atriumData.indices.length,
      array: atriumData.indices,
      getX: (i) => atriumData.indices[i],
    },
    attributes: {
      position: {
        count: atriumData.vertexCount,
        itemSize: 3,
        array: atriumData.positions,
        getX: (i) => atriumData.positions[i * 3],
        getY: (i) => atriumData.positions[i * 3 + 1],
        getZ: (i) => atriumData.positions[i * 3 + 2],
      },
    },
  },
};
const t0 = performance.now();
const bvh = buildContactBVH([mockAtriumMesh], MockTHREE, { gridSize: 1.0, range: 0.25 });
const bvhMs = performance.now() - t0;
assert.equal(bvh.triangles, atriumData.triangleCount);
assert.ok(bvh.nodes > 0);
assert.ok(bvh.bytes > 0);
assert.ok(bvh.uniforms.contactNodes.value);
console.log(`   OK (${bvh.triangles.toLocaleString()} tris, ${bvh.nodes.toLocaleString()} nodes, ${(bvh.bytes / 1024).toFixed(1)} KiB in ${bvhMs.toFixed(1)} ms)`);

console.log('6. Testing bin/bake-sdf.mjs --help and small OBJ bake...');
const helpOut = execFileSync('node', [path.join(repoRoot, 'bin/bake-sdf.mjs'), '--help'], { encoding: 'utf8' });
assert.ok(helpOut.includes('npx bake-sdf'));
assert.ok(helpOut.includes('--validate'));

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'fast-sdf-ao-test-'));
try {
  // Unit octahedron OBJ (closed watertight non-box mesh)
  const objPath = path.join(tmpDir, 'octa.obj');
  const outPath = path.join(tmpDir, 'octa.f16');
  fs.writeFileSync(objPath, [
    'v 1 0 0', 'v -1 0 0', 'v 0 1 0', 'v 0 -1 0', 'v 0 0 1', 'v 0 0 -1',
    'f 1 3 5', 'f 3 2 5', 'f 2 4 5', 'f 4 1 5',
    'f 3 1 6', 'f 2 3 6', 'f 4 2 6', 'f 1 4 6',
  ].join('\n') + '\n');
  const bakeOut = execFileSync('node', [
    path.join(repoRoot, 'bin/bake-sdf.mjs'),
    '--obj', objPath,
    '--lo', '-2', '-2', '-2',
    '--span', '4', '4', '4',
    '--res', '32',
    '--f16',
    '--validate', '200',
    '--out', outPath,
  ], { encoding: 'utf8' });
  const stats = JSON.parse(bakeOut.trim());
  assert.equal(stats.validate.mismatches, 0);
  assert.equal(stats.validate.sign_errors, 0);
  assert.equal(fs.statSync(outPath).size, 32 * 32 * 32 * 2);
  console.log(`   OK (32^3 FP16 bake in ${stats.ms.total.toFixed(1)} ms, 200/200 brute-force exact)`);
} finally {
  fs.rmSync(tmpDir, { recursive: true, force: true });
}

console.log('\nAll package Node tests PASSED!');
