// Unified WebGPU Compute Backend for Fast-SDF-AO:
// - Native WGSL storage-buffer Dual-Child Quantized SAH Contact BVH2 + 1-Lipschitz empty-sphere skip
// - 8x8 Workgroup Shared-Memory 2x2 Occlusion-Point Reuse (Jin et al. 2026)
// - SHaRC World-Space Spatial Hash Cache (262,144 atomic slots caching AO + Multi-Bounce GI, LRU eviction, converged-cell skip)
// - WGSL Compute GPU 64x64x40 Radiance Grid Relight + 2-Bounce Propagation + Floating Emissive Orb
// - 2-Pass Compute Edge-Preserving Hit-Distance-Guided A-Trous Denoiser + Full-Res Tangent-Plane Composite

export const WEBGPU_QUALITY_PRESETS = {
  low:    { label: 'Performance (Low)', rays: 1, maxSamples: 32, denoisePasses: 1, giMaxDist: 6.5, shadowSteps: 32, multiBounce: false },
  medium: { label: 'Balanced (Medium)', rays: 1, maxSamples: 48, denoisePasses: 2, giMaxDist: 7.5, shadowSteps: 44, multiBounce: true },
  high:   { label: 'High (Default)',    rays: 2, maxSamples: 64, denoisePasses: 2, giMaxDist: 8.5, shadowSteps: 56, multiBounce: true },
  ultra:  { label: 'Ultra',             rays: 4, maxSamples: 96, denoisePasses: 2, giMaxDist: 9.5, shadowSteps: 68, multiBounce: true },
};

const HASH_SLOTS = 262144; // 2^18 entries x 28 B = 7.0 MiB
const HASH_MASK = HASH_SLOTS - 1;
const RAD_DIMS = [64, 64, 40];
const RAD_SCALE = 3.5;

const GBUFFER_WGSL = /* wgsl */`
struct CamU {
  viewProj: mat4x4<f32>,
  view: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> cam: CamU;

struct VIn {
  @location(0) pos: vec3<f32>,
  @location(1) nor: vec3<f32>,
  @location(2) col: vec3<f32>,
};
struct VOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) wPos: vec3<f32>,
  @location(1) wNor: vec3<f32>,
  @location(2) col: vec3<f32>,
  @location(3) viewZ: f32,
};
@vertex fn vs(v: VIn) -> VOut {
  var o: VOut;
  let wp = vec4<f32>(v.pos, 1.0);
  o.clip = cam.viewProj * wp;
  o.wPos = v.pos;
  o.wNor = v.nor;
  o.col = v.col;
  o.viewZ = -(cam.view * wp).z;
  return o;
}
struct GOut {
  @location(0) posDepth: vec4<f32>,
  @location(1) normal: vec4<f32>,
  @location(2) albedo: vec4<f32>,
};
@fragment fn fs(i: VOut) -> GOut {
  var o: GOut;
  o.posDepth = vec4<f32>(i.wPos, i.viewZ);
  o.normal = vec4<f32>(normalize(i.wNor), 1.0);
  o.albedo = vec4<f32>(i.col, 1.0);
  return o;
}
`;

const RELIGHT_WGSL = /* wgsl */`
struct RelightU {
  sdfMin: vec4<f32>,
  sdfInvSpan: vec4<f32>,
  sunDir: vec4<f32>,      // xyz=dir, w=radScale
  emissivePos: vec4<f32>, // xyz=pos, w=radius
  emissiveCol: vec4<f32>, // rgb=color, w=bounceWeight
  dims: vec4<u32>,        // xyz=radDims, w=pass(0=direct,1=bounce)
};
@group(0) @binding(0) var<uniform> u: RelightU;
@group(0) @binding(1) var sdfTex: texture_3d<f32>;
@group(0) @binding(2) var linSamp: sampler;
@group(0) @binding(3) var albedoVol: texture_3d<f32>;
@group(0) @binding(4) var normalVol: texture_3d<f32>;
@group(0) @binding(5) var prevRadVol: texture_3d<f32>;
@group(0) @binding(6) var outRadVol: texture_storage_3d<rgba8unorm, write>;

fn sampleSDF(p: vec3<f32>) -> f32 {
  let uv = (p - u.sdfMin.xyz) * u.sdfInvSpan.xyz;
  if (any(uv < vec3<f32>(0.0)) || any(uv > vec3<f32>(1.0))) { return 1e4; }
  return textureSampleLevel(sdfTex, linSamp, uv, 0.0).r;
}

fn voxelCenter(c: vec3<i32>) -> vec3<f32> {
  let uv = (vec3<f32>(c) + 0.5) / vec3<f32>(u.dims.xyz);
  return u.sdfMin.xyz + uv / u.sdfInvSpan.xyz;
}

fn decodeDilOffset(codeF: f32) -> vec3<i32> {
  let code = i32(round(codeF * 255.0));
  if (code <= 0 || code > 27) { return vec3<i32>(0); }
  let c = code - 1;
  return vec3<i32>((c % 3) - 1, ((c / 3) % 3) - 1, (c / 9) - 1);
}

fn sdfSunSh(p: vec3<f32>, n: vec3<f32>, sDir: vec3<f32>) -> f32 {
  let ro = p + n * 0.06 + sDir * 0.04;
  var t = 0.04;
  var res = 1.0;
  for (var s = 0u; s < 20u; s++) {
    let d = sampleSDF(ro + sDir * t);
    if (d > 100.0) { break; }
    if (d < 0.012) { return 0.0; }
    res = min(res, 8.0 * d / t);
    t += max(d * 0.85, 0.06);
    if (t > 9.0) { break; }
  }
  return clamp(res, 0.0, 1.0);
}

fn sdfPtSh(p: vec3<f32>, n: vec3<f32>, lp: vec3<f32>, rLim: f32) -> f32 {
  let toL = lp - p;
  let dist = length(toL);
  if (dist < rLim + 0.04) { return 1.0; }
  let lDir = toL / dist;
  let ro = p + n * 0.05 + lDir * 0.03;
  var t = 0.03;
  let tMax = dist - rLim - 0.03;
  var res = 1.0;
  for (var s = 0u; s < 16u; s++) {
    let d = sampleSDF(ro + lDir * t);
    if (d < 0.01) { return 0.0; }
    res = min(res, 6.0 * d / t);
    t += max(d * 0.85, 0.05);
    if (t > tMax) { break; }
  }
  return clamp(res, 0.0, 1.0);
}

@compute @workgroup_size(4, 4, 4)
fn relightDirect(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= u.dims.xyz)) { return; }
  let coord = vec3<i32>(gid);
  let alb = textureLoad(albedoVol, coord, 0);
  let norEnc = textureLoad(normalVol, coord, 0);
  let p = voxelCenter(coord);

  // Inject floating emissive orb directly into open-air voxels overlapping orb sphere
  let eCol = u.emissiveCol.rgb;
  let hasOrb = dot(eCol, eCol) > 1e-4;
  if (hasOrb) {
    let distOrb = length(p - u.emissivePos.xyz);
    if (distOrb < u.emissivePos.w + 0.14) {
      let fall = smoothstep(u.emissivePos.w + 0.14, u.emissivePos.w * 0.5, distOrb);
      let enc = clamp((eCol * 2.2 * fall) / u.sunDir.w, vec3<f32>(0.0), vec3<f32>(1.0));
      textureStore(outRadVol, coord, vec4<f32>(enc, 1.0));
      return;
    }
  }

  var srcCoord = coord;
  var occupied = alb.a > 0.5;
  if (!occupied) {
    let off = decodeDilOffset(norEnc.a);
    if (any(off != vec3<i32>(0))) {
      srcCoord = clamp(coord + off, vec3<i32>(0), vec3<i32>(u.dims.xyz) - 1);
      occupied = true;
    }
  }
  if (!occupied) {
    textureStore(outRadVol, coord, vec4<f32>(0.0));
    return;
  }

  let sAlb = textureLoad(albedoVol, srcCoord, 0).rgb;
  let sNor = normalize(textureLoad(normalVol, srcCoord, 0).xyz * 2.0 - 1.0);
  let sp = voxelCenter(srcCoord);
  let sDir = normalize(u.sunDir.xyz);
  let ndl = max(0.0, dot(sNor, sDir));
  var sunSh = 0.0;
  if (ndl > 0.001) { sunSh = sdfSunSh(sp, sNor, sDir); }
  var rad = sAlb * (vec3<f32>(2.35, 2.20, 1.95) * ndl * sunSh + vec3<f32>(0.055, 0.065, 0.08));

  if (hasOrb) {
    let toO = u.emissivePos.xyz - sp;
    let d2 = dot(toO, toO);
    let dist = sqrt(d2);
    let lDir = toO / max(dist, 1e-4);
    let ndlO = max(0.0, dot(sNor, lDir));
    if (ndlO > 0.001 && dist < 7.0) {
      let att = 1.0 / (0.32 + d2);
      let oSh = sdfPtSh(sp, sNor, u.emissivePos.xyz, u.emissivePos.w);
      rad += sAlb * eCol * (2.8 * att * ndlO * oSh);
    }
  }

  let enc = clamp(rad / u.sunDir.w, vec3<f32>(0.0), vec3<f32>(1.0));
  textureStore(outRadVol, coord, vec4<f32>(enc, 1.0));
}

@compute @workgroup_size(4, 4, 4)
fn relightBounce(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= u.dims.xyz)) { return; }
  let coord = vec3<i32>(gid);
  let base = textureLoad(prevRadVol, coord, 0);
  if (base.a < 0.5) {
    textureStore(outRadVol, coord, vec4<f32>(0.0));
    return;
  }
  let alb0 = textureLoad(albedoVol, coord, 0);
  let nor0 = textureLoad(normalVol, coord, 0);
  var srcCoord = coord;
  if (alb0.a < 0.5) {
    let off = decodeDilOffset(nor0.a);
    srcCoord = clamp(coord + off, vec3<i32>(0), vec3<i32>(u.dims.xyz) - 1);
  }
  let sAlb = textureLoad(albedoVol, srcCoord, 0).rgb;
  let n = normalize(textureLoad(normalVol, srcCoord, 0).xyz * 2.0 - 1.0);
  let sp = voxelCenter(srcCoord);

  let ax = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.x) < 0.9);
  let tVec = normalize(cross(n, ax));
  let bVec = cross(n, tVec);
  let dirs = array<vec3<f32>, 6>(
    n,
    normalize(n * 0.55 + tVec * 0.835),
    normalize(n * 0.55 - tVec * 0.835),
    normalize(n * 0.55 + bVec * 0.835),
    normalize(n * 0.55 - bVec * 0.835),
    normalize(n * 0.8 + (tVec + bVec) * 0.42)
  );
  var ind = vec3<f32>(0.0);
  for (var i = 0u; i < 6u; i++) {
    let w = dirs[i];
    var dist = 0.25;
    for (var k = 0u; k < 6u; k++) {
      let q = sp + w * dist;
      let d = sampleSDF(q);
      if (d > 100.0) { break; }
      if (d < 0.08) {
        let uv = (q - u.sdfMin.xyz) * u.sdfInvSpan.xyz;
        if (all(uv >= vec3<f32>(0.0)) && all(uv <= vec3<f32>(1.0))) {
          ind += textureSampleLevel(prevRadVol, linSamp, uv, 0.0).rgb * u.sunDir.w;
        }
        break;
      }
      dist += max(d * 0.9, 0.22);
      if (dist > 5.0) { break; }
    }
  }
  let direct = base.rgb * u.sunDir.w;
  let total = direct + sAlb * (ind * (u.emissiveCol.w / 6.0));
  let enc = clamp(total / u.sunDir.w, vec3<f32>(0.0), vec3<f32>(1.0));
  textureStore(outRadVol, coord, vec4<f32>(enc, 1.0));
}
`;

const HASH_DECAY_WGSL = /* wgsl */`
struct HashEntry {
  key: atomic<u32>,
  aoSum: atomic<u32>,
  giR: atomic<u32>,
  giG: atomic<u32>,
  giB: atomic<u32>,
  count: atomic<u32>,
  lastFrame: atomic<u32>,
};
struct DecayU {
  params: vec4<u32>, // x=slots, y=frame, z=decayShift(0=none,1=soft), w=clearAll
};
@group(0) @binding(0) var<uniform> u: DecayU;
@group(0) @binding(1) var<storage, read_write> hashTable: array<HashEntry>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= u.params.x) { return; }
  if (u.params.w != 0u) {
    atomicStore(&hashTable[idx].key, 0u);
    atomicStore(&hashTable[idx].aoSum, 0u);
    atomicStore(&hashTable[idx].giR, 0u);
    atomicStore(&hashTable[idx].giG, 0u);
    atomicStore(&hashTable[idx].giB, 0u);
    atomicStore(&hashTable[idx].count, 0u);
    atomicStore(&hashTable[idx].lastFrame, 0u);
    return;
  }
  let k = atomicLoad(&hashTable[idx].key);
  if (k == 0u) { return; }
  let cnt = atomicLoad(&hashTable[idx].count);
  let age = atomicLoad(&hashTable[idx].lastFrame);
  if (u.params.y > age + 240u) {
    atomicStore(&hashTable[idx].key, 0u);
    atomicStore(&hashTable[idx].count, 0u);
    return;
  }
  // Soft lighting-change decay: keep ~75% weight capped at 6 samples so moving sun/orb
  // updates smoothly with zero black flash and zero flicker.
  if (u.params.z != 0u && cnt > 6u) {
    let targetCnt = 6u;
    let aoOld = atomicLoad(&hashTable[idx].aoSum);
    let rOld = atomicLoad(&hashTable[idx].giR);
    let gOld = atomicLoad(&hashTable[idx].giG);
    let bOld = atomicLoad(&hashTable[idx].giB);
    atomicStore(&hashTable[idx].aoSum, (aoOld * targetCnt) / cnt);
    atomicStore(&hashTable[idx].giR, (rOld * targetCnt) / cnt);
    atomicStore(&hashTable[idx].giG, (gOld * targetCnt) / cnt);
    atomicStore(&hashTable[idx].giB, (bOld * targetCnt) / cnt);
    atomicStore(&hashTable[idx].count, targetCnt);
  }
}
`;

const TRACE_WGSL = /* wgsl */`
struct TraceU {
  sdfMin: vec4<f32>,
  sdfInvSpan: vec4<f32>,
  gridMinSize: vec4<f32>,   // xyz=gridMin, w=gridSize
  gridDimsRange: vec4<f32>, // xyz=gridDims, w=contactRange
  sunDirScale: vec4<f32>,   // xyz=sunDir, w=radScale
  emissivePos: vec4<f32>,   // xyz=pos, w=radius
  emissiveCol: vec4<f32>,   // rgb=col, w=giMaxDist
  params: vec4<f32>,        // x=W, y=H, z=frame, w=contactBias
  flags: vec4<u32>,         // x=rays, y=maxSamples, z=useGI(0/1), w=useHash(0/1)
  extra: vec4<u32>,         // x=shadowSteps, y=tableMask, z=reuseEnabled, w=bvhNodeCount
  rootMin: vec4<f32>,
  rootSpan: vec4<f32>,
};

struct HashEntry {
  key: atomic<u32>,
  aoSum: atomic<u32>,
  giR: atomic<u32>,
  giG: atomic<u32>,
  giB: atomic<u32>,
  count: atomic<u32>,
  lastFrame: atomic<u32>,
};

struct HitRec {
  posDist: vec4<f32>, // xyz=hitPos, w=hitDist (-1 if miss)
  norValid: vec4<f32>,// xyz=hitNormal, w=1 if valid hit
  radVis: vec4<f32>,  // rgb=hitRad, w=rayVis
};

@group(0) @binding(0) var<uniform> u: TraceU;
@group(0) @binding(1) var gPosDepth: texture_2d<f32>;
@group(0) @binding(2) var gNormal: texture_2d<f32>;
@group(0) @binding(3) var sdfTex: texture_3d<f32>;
@group(0) @binding(4) var radVol: texture_3d<f32>;
@group(0) @binding(5) var linSamp: sampler;
@group(0) @binding(6) var<storage, read> bvhNodes: array<vec4<u32>>;
@group(0) @binding(7) var<storage, read> bvhTris: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> bvhGrid: array<vec2<u32>>;
@group(0) @binding(9) var<storage, read_write> hashTable: array<HashEntry>;
@group(0) @binding(10) var outAOGI: texture_storage_2d<rgba16float, write>;
@group(0) @binding(11) var outMeta: texture_storage_2d<rgba16float, write>;
@group(0) @binding(12) var<storage, read_write> stats: array<atomic<u32>>;

var<workgroup> sharedHits: array<HitRec, 64>;

fn ign(q: vec2<f32>) -> f32 {
  return fract(52.9829189 * fract(dot(q, vec2<f32>(0.06711056, 0.00583715))));
}

fn vdc2(bits_in: u32) -> f32 {
  var b = bits_in;
  b = (b << 16u) | (b >> 16u);
  b = ((b & 0x55555555u) << 1u) | ((b & 0xAAAAAAAAu) >> 1u);
  b = ((b & 0x33333333u) << 2u) | ((b & 0xCCCCCCCCu) >> 2u);
  b = ((b & 0x0F0F0F0Fu) << 4u) | ((b & 0xF0F0F0F0u) >> 4u);
  b = ((b & 0x00FF00FFu) << 8u) | ((b & 0xFF00FF00u) >> 8u);
  return f32(b) * 2.3283064365386963e-10;
}

fn vdc3(n_in: u32) -> f32 {
  var n = n_in;
  var f = 1.0;
  var r = 0.0;
  for (var j = 0u; j < 6u; j++) {
    f /= 3.0;
    r += f * f32(n % 3u);
    n /= 3u;
  }
  return r;
}

fn pcg(v_in: u32) -> u32 {
  var state = v_in * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn hashCell(cx: i32, cy: i32, cz: i32, nCode: u32, lod: u32) -> vec2<u32> {
  let ux = bitcast<u32>(cx);
  let uy = bitcast<u32>(cy);
  let uz = bitcast<u32>(cz);
  let h1 = pcg(ux ^ pcg(uy ^ pcg(uz ^ pcg((nCode << 5u) | lod))));
  var h2 = pcg(uz ^ pcg(ux ^ pcg(uy ^ pcg(((lod << 12u) | nCode) ^ 0x9e3779b9u))));
  if (h2 == 0u) { h2 = 1u; }
  return vec2<u32>(h1, h2);
}

fn sampleSDF(p: vec3<f32>) -> f32 {
  let uv = (p - u.sdfMin.xyz) * u.sdfInvSpan.xyz;
  if (any(uv < vec3<f32>(0.0)) || any(uv > vec3<f32>(1.0))) { return 1e4; }
  return textureSampleLevel(sdfTex, linSamp, uv, 0.0).r;
}

fn sampleRad(p: vec3<f32>) -> vec3<f32> {
  let uv = (p - u.sdfMin.xyz) * u.sdfInvSpan.xyz;
  if (any(uv < vec3<f32>(0.0)) || any(uv > vec3<f32>(1.0))) { return vec3<f32>(0.0); }
  return textureSampleLevel(radVol, linSamp, uv, 0.0).rgb * u.sunDirScale.w;
}

fn boxHit(nd: vec4<u32>, qInv: vec3<f32>, qOff: vec3<f32>, tMax: f32) -> f32 {
  let lo = vec3<f32>(f32(nd.x & 0xFFFFu), f32(nd.x >> 16u), f32(nd.y & 0xFFFFu));
  let hi = vec3<f32>(f32(nd.y >> 16u), f32(nd.z & 0xFFFFu), f32(nd.z >> 16u));
  let a = lo * qInv + qOff;
  let b = hi * qInv + qOff;
  let nearT = min(a, b);
  let farT = max(a, b);
  let t0 = max(0.0, max(nearT.x, max(nearT.y, nearT.z)));
  let t1 = min(tMax, min(farT.x, min(farT.y, farT.z)));
  return select(-1.0, t0, t1 >= t0);
}

fn testLeafHit(leafMeta: u32, origin: vec3<f32>, w: vec3<f32>, bestT: ptr<function, f32>, bestN: ptr<function, vec3<f32>>) -> bool {
  let first = (leafMeta & 0x7FFFFFFFu) >> 3u;
  let count = leafMeta & 7u;
  var hit = false;
  for (var j = 0u; j < 7u; j++) {
    if (j >= count) { break; }
    let tri = (first + j) * 3u;
    let v0 = bvhTris[tri].xyz;
    let e1 = bvhTris[tri + 1u].xyz;
    let e2 = bvhTris[tri + 2u].xyz;
    let h = cross(w, e2);
    let det = dot(e1, h);
    if (abs(det) < 1e-10) { continue; }
    let invDet = 1.0 / det;
    let s = origin - v0;
    let uu = dot(s, h) * invDet;
    if (uu < 0.0 || uu > 1.0) { continue; }
    let q = cross(s, e1);
    let vv = dot(w, q) * invDet;
    if (vv < 0.0 || uu + vv > 1.0) { continue; }
    let dist = dot(e2, q) * invDet;
    if (dist > 0.0 && dist < *bestT) {
      *bestT = dist;
      *bestN = cross(e1, e2);
      hit = true;
    }
  }
  return hit;
}

fn bvhContactHit(origin: vec3<f32>, w: vec3<f32>, root: u32, cnt: u32, qMin: vec3<f32>, qScale: vec3<f32>, bestT: ptr<function, f32>, bestN: ptr<function, vec3<f32>>) -> bool {
  *bestT = u.gridDimsRange.w;
  if (cnt == 0u) { return false; }
  let inv = vec3<f32>(1.0) / vec3<f32>(
    select(w.x, 1e-12, abs(w.x) < 1e-12),
    select(w.y, 1e-12, abs(w.y) < 1e-12),
    select(w.z, 1e-12, abs(w.z) < 1e-12)
  );
  let qInv = qScale * inv;
  let qOff = (qMin - origin) * inv;
  if (cnt == 1u) {
    let nd = bvhNodes[root];
    if (boxHit(nd, qInv, qOff, *bestT) < 0.0) { return false; }
    return testLeafHit(nd.w, origin, w, bestT, bestN);
  }
  var anyHit = false;
  var stack: array<u32, 16>;
  var sp = 0u;
  var pair = root;
  for (var visit = 0u; visit < 512u; visit++) {
    let ndL = bvhNodes[pair];
    let ndR = bvhNodes[pair + 1u];
    let tL = boxHit(ndL, qInv, qOff, *bestT);
    let tR = boxHit(ndR, qInv, qOff, *bestT);
    var hitL = tL >= 0.0 && tL < *bestT;
    var hitR = tR >= 0.0 && tR < *bestT;
    if (hitL && (ndL.w & 0x80000000u) != 0u) {
      if (testLeafHit(ndL.w, origin, w, bestT, bestN)) { anyHit = true; }
      hitL = false;
    }
    if (hitR && (ndR.w & 0x80000000u) != 0u) {
      if (testLeafHit(ndR.w, origin, w, bestT, bestN)) { anyHit = true; }
      hitR = false;
    }
    if (hitL && hitR) {
      if (tL <= tR) {
        if (sp < 16u) { stack[sp] = ndR.w; sp += 1u; }
        pair = ndL.w;
      } else {
        if (sp < 16u) { stack[sp] = ndL.w; sp += 1u; }
        pair = ndR.w;
      }
      continue;
    }
    if (hitL) { pair = ndL.w; continue; }
    if (hitR) { pair = ndR.w; continue; }
    if (sp == 0u) { break; }
    sp -= 1u;
    pair = stack[sp];
  }
  return anyHit;
}

fn sdfSunShadow(pos: vec3<f32>, nor: vec3<f32>, sDir: vec3<f32>) -> f32 {
  let ndl = dot(nor, sDir);
  if (ndl <= 0.001) { return 0.0; }
  let ro = pos + nor * 0.03 + sDir * 0.02;
  var t = 0.02;
  var res = 1.0;
  let steps = u.extra.x;
  for (var s = 0u; s < steps; s++) {
    let d = sampleSDF(ro + sDir * t);
    if (d > 100.0) { break; }
    let tol = min(0.012, 0.22 * t * max(ndl, 0.15));
    let dc = d - tol;
    if (dc <= 0.0) { return 0.0; }
    res = min(res, 8.0 * dc / t);
    if (res < 0.005) { return 0.0; }
    let k = mix(0.4, 0.8, smoothstep(0.40, 0.85, d));
    t += max(d * k, 0.014);
    if (t > 10.5) { break; }
  }
  res = clamp(res, 0.0, 1.0);
  return res * res * (3.0 - 2.0 * res);
}

fn sdfPointShadow(pos: vec3<f32>, nor: vec3<f32>, lp: vec3<f32>, rLim: f32) -> f32 {
  let toL = lp - pos;
  let dist = length(toL);
  if (dist < rLim + 0.03 || dist > 7.5) { return 1.0; }
  let lDir = toL / dist;
  if (dot(nor, lDir) <= 0.001) { return 0.0; }
  let ro = pos + nor * 0.025 + lDir * 0.015;
  var t = 0.015;
  let tMax = dist - rLim - 0.025;
  var res = 1.0;
  for (var s = 0u; s < 32u; s++) {
    let d = sampleSDF(ro + lDir * t);
    if (d < 0.008) { return 0.0; }
    res = min(res, 6.0 * d / t);
    if (res < 0.005) { return 0.0; }
    t += max(d * 0.75, 0.018);
    if (t > tMax) { break; }
  }
  res = clamp(res, 0.0, 1.0);
  return res * res * (3.0 - 2.0 * res);
}

@compute @workgroup_size(8, 8)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(local_invocation_index) lidx: u32
) {
  let W = u32(u.params.x);
  let H = u32(u.params.y);
  let validPx = gid.x < W && gid.y < H;

  // Initialize workgroup shared slot
  sharedHits[lidx] = HitRec(vec4<f32>(0.0, 0.0, 0.0, -1.0), vec4<f32>(0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0));

  var p = vec3<f32>(0.0);
  var n = vec3<f32>(0.0, 0.0, 1.0);
  var viewDepth = 0.0;
  var isSurf = false;

  if (validPx) {
    let pd = textureLoad(gPosDepth, vec2<i32>(gid.xy), 0);
    let gn = textureLoad(gNormal, vec2<i32>(gid.xy), 0);
    viewDepth = pd.w;
    if (viewDepth > 0.0 && gn.w > 0.5) {
      p = pd.xyz;
      n = normalize(gn.xyz);
      isSurf = true;
    }
  }

  let frame = u32(u.params.z);
  let px = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);
  let useGI = u.flags.z != 0u;
  let useHash = u.flags.w != 0u;
  let maxSamples = u.flags.y;
  let tableMask = u.extra.y;

  // 1) SHaRC World-Space Spatial Hash lookup (distance-LOD + tangent-plane jitter)
  var slotFound = 0xFFFFFFFFu;
  var cacheHit = false;
  var cachedAO = 1.0;
  var cachedGI = vec3<f32>(0.0);
  var cachedCount = 0u;

  if (isSurf && useHash) {
    let lod = u32(clamp(floor(log2(max(viewDepth * 0.25, 0.5))), 0.0, 4.0));
    let cellSize = 0.028 * exp2(f32(lod));
    let ax = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.x) < 0.9);
    let tang = normalize(cross(n, ax));
    let bitang = cross(n, tang);
    let jx = (ign(px + vec2<f32>(11.0, 23.0)) - 0.5) * cellSize;
    let jy = (ign(px + vec2<f32>(41.0, 7.0)) - 0.5) * cellSize;
    let pj = p + tang * jx + bitang * jy;
    let cx = i32(floor(pj.x / cellSize));
    let cy = i32(floor(pj.y / cellSize));
    let cz = i32(floor(pj.z / cellSize));
    let nxQ = u32(clamp(floor((n.x * 0.5 + 0.5) * 3.99), 0.0, 3.0));
    let nyQ = u32(clamp(floor((n.y * 0.5 + 0.5) * 3.99), 0.0, 3.0));
    let nzQ = u32(clamp(floor((n.z * 0.5 + 0.5) * 3.99), 0.0, 3.0));
    let nCode = nxQ | (nyQ << 2u) | (nzQ << 4u);
    let h = hashCell(cx, cy, cz, nCode, lod);
    let baseSlot = h.x & tableMask;
    let key = h.y;

    for (var probe = 0u; probe < 12u; probe++) {
      let sIdx = (baseSlot + probe) & tableMask;
      let existing = atomicLoad(&hashTable[sIdx].key);
      if (existing == key) {
        slotFound = sIdx;
        cachedCount = atomicLoad(&hashTable[sIdx].count);
        atomicStore(&hashTable[sIdx].lastFrame, frame);
        if (cachedCount > 0u) {
          let invC = 1.0 / (f32(cachedCount) * 65535.0);
          cachedAO = clamp(f32(atomicLoad(&hashTable[sIdx].aoSum)) * invC, 0.0, 1.0);
          cachedGI = vec3<f32>(
            f32(atomicLoad(&hashTable[sIdx].giR)),
            f32(atomicLoad(&hashTable[sIdx].giG)),
            f32(atomicLoad(&hashTable[sIdx].giB))
          ) * (u.sunDirScale.w * invC);
          if (cachedCount >= maxSamples) { cacheHit = true; }
        }
        break;
      }
      if (existing == 0u) {
        let res = atomicCompareExchangeWeak(&hashTable[sIdx].key, 0u, key);
        if (res.exchanged || res.old_value == key) {
          slotFound = sIdx;
          if (res.exchanged) {
            atomicStore(&hashTable[sIdx].aoSum, 0u);
            atomicStore(&hashTable[sIdx].giR, 0u);
            atomicStore(&hashTable[sIdx].giG, 0u);
            atomicStore(&hashTable[sIdx].giB, 0u);
            atomicStore(&hashTable[sIdx].count, 0u);
            atomicStore(&hashTable[sIdx].lastFrame, frame);
          }
          break;
        }
      }
    }
  }

  // 2) Trace AO + 1-Bounce/Multi-Bounce GI if not converged in SHaRC cache
  var tracedAO = cachedAO;
  var tracedGI = cachedGI;
  var avgHitDist = 1.12;
  let needTrace = isSurf && !cacheHit;

  if (needTrace) {
    let nd = u.flags.x;
    let ax = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.x) < 0.9);
    let tVec = normalize(cross(n, ax));
    let bVec = cross(n, tVec);
    let su = ign(px);
    let sp = ign(px + vec2<f32>(37.0, 17.0));
    let fu = fract(su + vdc2(frame));
    let phiBase = 6.2831853 * fract(sp + vdc3(frame));

    let cRange = u.gridDimsRange.w;
    let origin = p + n * u.params.w;

    // 1-Lipschitz empty-sphere skip
    var minCosClear = 2.0;
    let dClear2 = sampleSDF(p + n * (2.0 * cRange));
    if (dClear2 >= 2.0 * cRange - 0.004 && dClear2 < 1e3) {
      minCosClear = 0.25;
    } else {
      let dClear1 = sampleSDF(p + n * cRange);
      if (dClear1 >= cRange - 0.003 && dClear1 < 1e3) { minCosClear = 0.5; }
    }

    // Hoisted 3D contact grid lookup (direct storage buffer)
    var cellRoot = 0u;
    var cellCnt = 0u;
    var cellQMin = u.rootMin.xyz;
    var cellQScale = u.rootSpan.xyz * (1.0 / 65535.0);
    let gSize = u.gridMinSize.w;
    if (gSize > 0.0) {
      let gDims = vec3<i32>(u.gridDimsRange.xyz);
      let cell = vec3<i32>(floor((p - u.gridMinSize.xyz) / gSize));
      if (all(cell >= vec3<i32>(0)) && all(cell < gDims)) {
        let cid = u32((cell.z * gDims.y + cell.y) * gDims.x + cell.x);
        let bnd = bvhGrid[cid];
        cellRoot = bnd.x;
        cellCnt = bnd.y;
        cellQMin = u.gridMinSize.xyz + vec3<f32>(cell) * gSize - vec3<f32>(cRange + abs(u.params.w));
        cellQScale = vec3<f32>((gSize + 2.0 * (cRange + abs(u.params.w))) * (1.0 / 65535.0));
      }
    } else {
      cellCnt = u.extra.w;
    }

    var visSum = 0.0;
    var giSum = vec3<f32>(0.0);
    var hitDistSum = 0.0;
    var bestShare = HitRec(vec4<f32>(0.0, 0.0, 0.0, -1.0), vec4<f32>(0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0));

    for (var i = 0u; i < 4u; i++) {
      if (i >= nd) { break; }
      let uu = clamp((f32(i) + fu) / f32(nd), 1e-4, 0.9999);
      let phi = phiBase + f32(i) * (6.2831853 / f32(nd));
      let cosn = sqrt(1.0 - uu);
      let sinT = sqrt(uu);
      let w = tVec * (sinT * cos(phi)) + bVec * (sinT * sin(phi)) + n * cosn;

      var tHit = cRange;
      var hitN = vec3<f32>(0.0);
      var blocked = false;
      if (cosn < minCosClear) {
        blocked = bvhContactHit(origin, w, cellRoot, cellCnt, cellQMin, cellQScale, &tHit, &hitN);
      }

      var v = 1.0;
      var rad = vec3<f32>(0.0);
      var rayHitT = u.emissiveCol.w;

      if (blocked) {
        v = 0.0;
        rayHitT = tHit;
        if (useGI) {
          let nl = length(hitN);
          if (nl > 1e-8) {
            let hn = hitN / nl;
            if (dot(hn, -w) > 0.0) {
              rad = sampleRad(origin + w * tHit);
              if (tHit < 0.15) {
                let sDir = normalize(u.sunDirScale.xyz);
                let hitNdl = max(0.0, dot(hn, sDir));
                rad *= mix(0.28 + 0.72 * smoothstep(0.0, 0.16, hitNdl), 1.0, smoothstep(0.04, 0.15, tHit));
              }
              if (i == 0u) {
                bestShare = HitRec(vec4<f32>(origin + w * tHit, tHit), vec4<f32>(hn, 1.0), vec4<f32>(rad, 0.0));
              }
            }
          }
        } else if (i == 0u) {
          let hn = select(vec3<f32>(0.0, 0.0, 1.0), normalize(hitN), length(hitN) > 1e-8);
          bestShare = HitRec(vec4<f32>(origin + w * tHit, tHit), vec4<f32>(hn, 1.0), vec4<f32>(0.0));
        }
      } else {
        var s = cRange;
        var minD = 1e4;
        var tMinD = cRange;
        for (var k = 0u; k < 16u; k++) {
          let d = sampleSDF(p + w * s);
          if (d < minD) { minD = d; tMinD = s; }
          let e = max(min(0.025, 0.1 * s * cosn), 1e-6);
          v = min(v, clamp((d + e) / (4.0 * e), 0.0, 1.0));
          if (v < 0.01) { break; }
          s += max(d, 0.072);
          if (s > 1.12) { break; }
        }
        if (useGI) {
          if (v < 0.5) {
            rayHitT = tMinD;
            rad = sampleRad(p + w * tMinD);
          } else {
            var sg = max(s, 1.12);
            var hitFar = false;
            for (var k = 0u; k < 14u; k++) {
              let d = sampleSDF(p + w * sg);
              if (d > 100.0) { break; }
              if (d < 0.045) {
                rayHitT = sg;
                rad = sampleRad(p + w * sg);
                hitFar = true;
                break;
              }
              sg += select(max(d, 0.08), d * 1.15, d > 0.30);
              if (sg > u.emissiveCol.w) { break; }
            }
            if (!hitFar) {
              // Subtle skylight contribution on open ceiling hits
              rad = vec3<f32>(0.18, 0.22, 0.28) * max(0.0, w.z);
            }
          }
          if (i == 0u && rayHitT < u.emissiveCol.w * 0.95) {
            bestShare = HitRec(vec4<f32>(p + w * rayHitT, rayHitT), vec4<f32>(-w, 1.0), vec4<f32>(rad, v));
          }
        } else if (i == 0u && v < 0.5) {
          bestShare = HitRec(vec4<f32>(p + w * tMinD, tMinD), vec4<f32>(-w, 1.0), vec4<f32>(0.0, 0.0, 0.0, v));
        }
      }
      visSum += v;
      giSum += rad;
      hitDistSum += rayHitT;
    }

    sharedHits[lidx] = bestShare;
    tracedAO = visSum / f32(nd);
    tracedGI = giSum / f32(nd);
    avgHitDist = hitDistSum / f32(nd);
  }

  // 3) Workgroup Shared-Memory 2x2 Occlusion-Point Reuse (Jin et al. 2026)
  workgroupBarrier();

  if (needTrace && u.extra.z != 0u) {
    let peerX = lidx ^ 1u;
    let peerY = lidx ^ 8u;
    var reuseVis = 0.0;
    var reuseGI = vec3<f32>(0.0);
    var reuseW = 0.0;
    let peers = array<u32, 2>(peerX, peerY);
    for (var pi = 0u; pi < 2u; pi++) {
      let hr = sharedHits[peers[pi]];
      if (hr.norValid.w > 0.5 && hr.posDist.w > 0.02) {
        let toH = hr.posDist.xyz - p;
        let dH = length(toH);
        if (dH > 0.03 && dH < u.emissiveCol.w) {
          let wH = toH / dH;
          let cosIn = dot(n, wH);
          let cosOut = dot(hr.norValid.xyz, -wH);
          if (cosIn > 0.08 && cosOut > 0.05) {
            let wt = clamp(cosIn * 0.35, 0.0, 0.25);
            if (dH <= 1.12) { reuseVis += hr.radVis.w * wt; } else { reuseVis += wt; }
            reuseGI += hr.radVis.rgb * wt;
            reuseW += wt;
          }
        }
      }
    }
    if (reuseW > 0.01) {
      tracedAO = (tracedAO + reuseVis) / (1.0 + reuseW);
      tracedGI = (tracedGI + reuseGI) / (1.0 + reuseW);
    }
  }

  // 4) Commit to SHaRC World-Space Spatial Hash Cache + output half-res targets
  if (!validPx) { return; }
  if (!isSurf) {
    textureStore(outAOGI, vec2<i32>(gid.xy), vec4<f32>(0.0, 0.0, 0.0, 1.0));
    textureStore(outMeta, vec2<i32>(gid.xy), vec4<f32>(1.0, 1.0, 1.0, 0.0));
    return;
  }

  var finalAO = tracedAO;
  var finalGI = tracedGI;
  var hitFlag = 0.0;

  if (cacheHit) {
    atomicAdd(&stats[1], 1u);
    hitFlag = 1.0;
  } else {
    atomicAdd(&stats[0], 1u);
    if (useHash && slotFound != 0xFFFFFFFFu) {
      let addAO = u32(clamp(tracedAO, 0.0, 1.0) * 65535.0);
      let invS = 65535.0 / u.sunDirScale.w;
      let addR = u32(clamp(tracedGI.r * invS, 0.0, 65535.0));
      let addG = u32(clamp(tracedGI.g * invS, 0.0, 65535.0));
      let addB = u32(clamp(tracedGI.b * invS, 0.0, 65535.0));
      let prevAO = atomicAdd(&hashTable[slotFound].aoSum, addAO);
      let prevR = atomicAdd(&hashTable[slotFound].giR, addR);
      let prevG = atomicAdd(&hashTable[slotFound].giG, addG);
      let prevB = atomicAdd(&hashTable[slotFound].giB, addB);
      let prevCnt = atomicAdd(&hashTable[slotFound].count, 1u);
      atomicStore(&hashTable[slotFound].lastFrame, frame);
      let newCnt = prevCnt + 1u;
      let invN = 1.0 / (f32(newCnt) * 65535.0);
      finalAO = clamp(f32(prevAO + addAO) * invN, 0.0, 1.0);
      finalGI = vec3<f32>(f32(prevR + addR), f32(prevG + addG), f32(prevB + addB)) * (u.sunDirScale.w * invN);
    }
  }

  // Half-res direct sun + orb shadows
  var sunSh = 1.0;
  var orbSh = 1.0;
  if (useGI) {
    sunSh = sdfSunShadow(p, n, normalize(u.sunDirScale.xyz));
    if (dot(u.emissiveCol.rgb, u.emissiveCol.rgb) > 1e-4) {
      orbSh = sdfPointShadow(p, n, u.emissivePos.xyz, u.emissivePos.w);
    }
  }

  let normHit = clamp(avgHitDist / 4.0, 0.05, 1.0);
  textureStore(outAOGI, vec2<i32>(gid.xy), vec4<f32>(finalGI, finalAO));
  textureStore(outMeta, vec2<i32>(gid.xy), vec4<f32>(sunSh, orbSh, normHit, hitFlag));
}
`;

const DENOISE_WGSL = /* wgsl */`
struct DenoiseU {
  params: vec4<u32>, // x=W, y=H, z=stepSize, w=filterAO(0/1)
};
@group(0) @binding(0) var<uniform> u: DenoiseU;
@group(0) @binding(1) var srcAOGI: texture_2d<f32>;
@group(0) @binding(2) var gPosDepth: texture_2d<f32>;
@group(0) @binding(3) var gNormal: texture_2d<f32>;
@group(0) @binding(4) var metaTex: texture_2d<f32>;
@group(0) @binding(5) var dstAOGI: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let W = i32(u.params.x);
  let H = i32(u.params.y);
  let q = vec2<i32>(gid.xy);
  if (q.x >= W || q.y >= H) { return; }

  let pdC = textureLoad(gPosDepth, q, 0);
  let gnC = textureLoad(gNormal, q, 0);
  let cC = textureLoad(srcAOGI, q, 0);
  if (pdC.w <= 0.0 || gnC.w < 0.5) {
    textureStore(dstAOGI, q, cC);
    return;
  }

  let pC = pdC.xyz;
  let nC = normalize(gnC.xyz);
  let metaC = textureLoad(metaTex, q, 0);
  let hitScale = clamp(sqrt(metaC.z) * 1.35, 0.45, 1.0);
  let stepS = max(1, i32(round(f32(u.params.z) * hitScale)));
  let planeTol = (0.0025 * pdC.w + 0.01) * f32(stepS);

  var giSum = cC.rgb;
  var aoSum = cC.a;
  var wSum = 1.0;

  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      if (dx == 0 && dy == 0) { continue; }
      let p = q + vec2<i32>(dx, dy) * stepS;
      if (p.x < 0 || p.y < 0 || p.x >= W || p.y >= H) { continue; }
      let pdP = textureLoad(gPosDepth, p, 0);
      let gnP = textureLoad(gNormal, p, 0);
      if (pdP.w <= 0.0 || gnP.w < 0.5) { continue; }
      let nP = normalize(gnP.xyz);
      if (dot(nP, nC) < 0.9) { continue; }
      if (abs(dot(nC, pdP.xyz - pC)) > planeTol) { continue; }
      let cP = textureLoad(srcAOGI, p, 0);
      let wK = select(0.25, 0.5, dx == 0 || dy == 0);
      giSum += cP.rgb * wK;
      aoSum += cP.a * wK;
      wSum += wK;
    }
  }
  let outAO = select(cC.a, aoSum / wSum, u.params.w != 0u);
  textureStore(dstAOGI, q, vec4<f32>(giSum / wSum, outAO));
}
`;

const COMPOSITE_WGSL = /* wgsl */`
struct CompU {
  viewProj: mat4x4<f32>,
  eye: vec4<f32>,
  sunDir: vec4<f32>,      // xyz=sunDir, w=useGI(0/1)
  emissivePos: vec4<f32>, // xyz=pos, w=radius
  emissiveCol: vec4<f32>, // rgb=col, w=viewMode(0=shaded,1=aoMask,2=giOnly,3=hashCache)
  screen: vec4<f32>,      // x=W, y=H, z=aoW, w=aoH
};
@group(0) @binding(0) var<uniform> u: CompU;
@group(0) @binding(1) var aoGITex: texture_2d<f32>;
@group(0) @binding(2) var gPosDepth: texture_2d<f32>;
@group(0) @binding(3) var gNormal: texture_2d<f32>;
@group(0) @binding(4) var metaTex: texture_2d<f32>;

struct VIn {
  @location(0) pos: vec3<f32>,
  @location(1) nor: vec3<f32>,
  @location(2) col: vec3<f32>,
};
struct VOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) wPos: vec3<f32>,
  @location(1) wNor: vec3<f32>,
  @location(2) col: vec3<f32>,
};
@vertex fn vs(v: VIn) -> VOut {
  var o: VOut;
  o.clip = u.viewProj * vec4<f32>(v.pos, 1.0);
  o.wPos = v.pos;
  o.wNor = v.nor;
  o.col = v.col;
  return o;
}

fn multiBounceAO(ao: f32, alb: vec3<f32>) -> vec3<f32> {
  let a = 2.0404 * alb - 0.3324;
  let b = -4.7951 * alb + 0.6417;
  let c = 2.7552 * alb + 0.6903;
  return max(vec3<f32>(ao), ((ao * a + b) * ao + c) * ao);
}

fn acesToneMap(x: vec3<f32>) -> vec3<f32> {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linearToSrgb(c: vec3<f32>) -> vec3<f32> {
  let lo = 12.92 * c;
  let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
  return select(lo, hi, c >= vec3<f32>(0.0031308));
}

@fragment fn fs(i: VOut) -> @location(0) vec4<f32> {
  let p = i.wPos;
  let n = normalize(i.wNor);
  let alb = i.col;
  let aoRes = u.screen.zw;
  let f = i.clip.xy * (aoRes / u.screen.xy) - 0.5;
  let i0 = vec2<i32>(floor(f));
  let fw = f - floor(f);
  let maxI = vec2<i32>(aoRes) - 1;

  let viewDepth = length(p - u.eye.xyz);
  let planeTol = 0.0025 * viewDepth + 0.01;

  var agSum = vec4<f32>(0.0);
  var mSum = vec4<f32>(0.0);
  var wSum = 0.0;
  var bestAG = vec4<f32>(0.0, 0.0, 0.0, 1.0);
  var bestM = vec4<f32>(1.0, 1.0, 1.0, 0.0);
  var bestScore = 1e9;

  for (var k = 0; k < 4; k++) {
    let o = vec2<i32>(k & 1, k >> 1);
    let q = clamp(i0 + o, vec2<i32>(0), maxI);
    let bw = select(1.0 - fw.x, fw.x, o.x == 1) * select(1.0 - fw.y, fw.y, o.y == 1);
    let pd = textureLoad(gPosDepth, q, 0);
    let gn = textureLoad(gNormal, q, 0);
    if (pd.w <= 0.0 || gn.w < 0.5) { continue; }
    let nQ = normalize(gn.xyz);
    let dn = dot(nQ, n);
    let pDist = abs(dot(n, pd.xyz - p));
    let ag = textureLoad(aoGITex, q, 0);
    let mt = textureLoad(metaTex, q, 0);
    let score = pDist + (1.0 - max(dn, 0.0)) * 0.5;
    if (score < bestScore) { bestScore = score; bestAG = ag; bestM = mt; }
    if (dn >= 0.88 && pDist <= planeTol) {
      agSum += ag * bw;
      mSum += mt * bw;
      wSum += bw;
    }
  }
  let ag = select(bestAG, agSum / wSum, wSum > 0.01);
  let mt = select(bestM, mSum / wSum, wSum > 0.01);
  let gi = ag.rgb;
  let ao = clamp(ag.a, 0.0, 1.0);
  let sunSh = mt.x;
  let orbSh = mt.y;
  let hashHit = mt.w;

  let mode = i32(round(u.emissiveCol.w));
  if (mode == 1) { return vec4<f32>(vec3<f32>(ao), 1.0); }
  if (mode == 2) { return vec4<f32>(linearToSrgb(acesToneMap(gi * 1.15)), 1.0); }
  if (mode == 3) {
    let dbg = mix(vec3<f32>(0.90, 0.20, 0.65), vec3<f32>(0.12, 0.85, 0.52), hashHit);
    return vec4<f32>(linearToSrgb(dbg * (0.35 + 0.65 * ao)), 1.0);
  }

  let sDir = normalize(u.sunDir.xyz);
  let sun = max(0.0, dot(n, sDir));
  if (u.sunDir.w < 0.5) {
    // Archive / AO-only shading
    let col = alb * (0.26 * sun + 0.74 * ao);
    return vec4<f32>(pow(max(col, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2)), 1.0);
  }

  // PBR + Multi-Bounce GI + Dynamic Emissive Orb
  let mbAO = multiBounceAO(ao, alb);
  let skyCol = vec3<f32>(0.68, 0.79, 0.96);
  let gndCol = vec3<f32>(0.30, 0.27, 0.24);
  let ambEnv = mix(gndCol, skyCol, n.z * 0.5 + 0.5);
  var color = alb * (vec3<f32>(2.25, 2.10, 1.85) * sun * sunSh + gi + ambEnv * mbAO * 0.16);

  if (dot(u.emissiveCol.rgb, u.emissiveCol.rgb) > 1e-4) {
    let toO = u.emissivePos.xyz - p;
    let d2 = dot(toO, toO);
    let dist = sqrt(d2);
    let lDir = toO / max(dist, 1e-4);
    let ndlO = max(0.0, dot(n, lDir));
    if (ndlO > 0.001 && dist < 7.5) {
      let att = 1.0 / (0.28 + d2);
      color += alb * u.emissiveCol.rgb * (2.6 * att * ndlO * orbSh);
    }
  }
  return vec4<f32>(linearToSrgb(acesToneMap(color * 0.88)), 1.0);
}
`;

// Minimal CPU triangle splatting into 64x64x40 albedo + normal (with 1-voxel dilation offset code in normal.a)
function buildStaticVoxelAttributes(positions, normals, colors, indices, sdfMin, sdfSpan, dims = RAD_DIMS) {
  const [nx, ny, nz] = dims;
  const total = nx * ny * nz;
  const accR = new Float32Array(total), accG = new Float32Array(total), accB = new Float32Array(total);
  const accNX = new Float32Array(total), accNY = new Float32Array(total), accNZ = new Float32Array(total);
  const accW = new Float32Array(total);
  const [minX, minY, minZ] = sdfMin;
  const [spanX, spanY, spanZ] = sdfSpan;
  const vxS = nx / spanX, vyS = ny / spanY, vzS = nz / spanZ;
  const stepDist = Math.min(spanX / nx, spanY / ny, spanZ / nz) * 0.82;
  const nTris = indices ? indices.length / 3 : positions.length / 9;

  for (let t = 0; t < nTris; t++) {
    const ia = indices ? indices[t * 3] : t * 3;
    const ib = indices ? indices[t * 3 + 1] : t * 3 + 1;
    const ic = indices ? indices[t * 3 + 2] : t * 3 + 2;
    const x0 = positions[ia * 3], y0 = positions[ia * 3 + 1], z0 = positions[ia * 3 + 2];
    const x1 = positions[ib * 3], y1 = positions[ib * 3 + 1], z1 = positions[ib * 3 + 2];
    const x2 = positions[ic * 3], y2 = positions[ic * 3 + 1], z2 = positions[ic * 3 + 2];
    const e1x = x1 - x0, e1y = y1 - y0, e1z = z1 - z0;
    const e2x = x2 - x0, e2y = y2 - y0, e2z = z2 - z0;
    const fnx = e1y * e2z - e1z * e2y, fny = e1z * e2x - e1x * e2z, fnz = e1x * e2y - e1y * e2x;
    const area2 = Math.hypot(fnx, fny, fnz);
    if (area2 < 1e-9) continue;
    const invA2 = 1 / area2;
    const unx = fnx * invA2, uny = fny * invA2, unz = fnz * invA2;
    const cr = (colors[ia * 3] + colors[ib * 3] + colors[ic * 3]) / 3;
    const cg = (colors[ia * 3 + 1] + colors[ib * 3 + 1] + colors[ic * 3 + 1]) / 3;
    const cb = (colors[ia * 3 + 2] + colors[ib * 3 + 2] + colors[ic * 3 + 2]) / 3;
    const maxEdge = Math.max(Math.hypot(e1x, e1y, e1z), Math.hypot(e2x, e2y, e2z), Math.hypot(x2 - x1, y2 - y1, z2 - z1));
    const steps = Math.max(1, Math.min(32, Math.ceil(maxEdge / stepDist)));
    const wPer = (0.5 * area2) / ((steps + 1) * (steps + 2) * 0.5);
    for (let u = 0; u <= steps; u++) {
      const fu = u / steps;
      for (let v = 0; v <= steps - u; v++) {
        const fv = v / steps;
        const px = x0 + e1x * fu + e2x * fv;
        const py = y0 + e1y * fu + e2y * fv;
        const pz = z0 + e1z * fu + e2z * fv;
        const ix = Math.floor((px - minX) * vxS);
        const iy = Math.floor((py - minY) * vyS);
        const iz = Math.floor((pz - minZ) * vzS);
        if (ix < 0 || iy < 0 || iz < 0 || ix >= nx || iy >= ny || iz >= nz) continue;
        const id = (iz * ny + iy) * nx + ix;
        accR[id] += cr * wPer; accG[id] += cg * wPer; accB[id] += cb * wPer;
        accNX[id] += unx * wPer; accNY[id] += uny * wPer; accNZ[id] += unz * wPer;
        accW[id] += wPer;
      }
    }
  }

  const albedoU8 = new Uint8Array(total * 4);
  const normalU8 = new Uint8Array(total * 4);
  for (let i = 0; i < total; i++) {
    const w = accW[i];
    if (w > 0) {
      const inv = 1 / w;
      albedoU8[i * 4]     = Math.min(255, Math.round(accR[i] * inv * 255));
      albedoU8[i * 4 + 1] = Math.min(255, Math.round(accG[i] * inv * 255));
      albedoU8[i * 4 + 2] = Math.min(255, Math.round(accB[i] * inv * 255));
      albedoU8[i * 4 + 3] = 255;
      const l = Math.hypot(accNX[i], accNY[i], accNZ[i]) || 1;
      normalU8[i * 4]     = Math.round((accNX[i] / l * 0.5 + 0.5) * 255);
      normalU8[i * 4 + 1] = Math.round((accNY[i] / l * 0.5 + 0.5) * 255);
      normalU8[i * 4 + 2] = Math.round((accNZ[i] / l * 0.5 + 0.5) * 255);
      normalU8[i * 4 + 3] = 0;
    }
  }
  // 1-voxel normal-aware dilation code into unoccupied neighbor cells
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const id = (z * ny + y) * nx + x;
        if (accW[id] > 0) continue;
        let bestScore = -1e9, bestCode = 0;
        for (let dz = -1; dz <= 1; dz++) {
          const nz0 = z + dz; if (nz0 < 0 || nz0 >= nz) continue;
          for (let dy = -1; dy <= 1; dy++) {
            const ny0 = y + dy; if (ny0 < 0 || ny0 >= ny) continue;
            for (let dx = -1; dx <= 1; dx++) {
              const nx0 = x + dx; if (nx0 < 0 || nx0 >= nx) continue;
              if (dx === 0 && dy === 0 && dz === 0) continue;
              const nid = (nz0 * ny + ny0) * nx + nx0;
              if (accW[nid] <= 0) continue;
              const l = Math.hypot(accNX[nid], accNY[nid], accNZ[nid]) || 1;
              const dotDir = -(accNX[nid] * dx + accNY[nid] * dy + accNZ[nid] * dz) / (l * Math.hypot(dx, dy, dz));
              const score = dotDir * 2.0 - (Math.abs(dx) + Math.abs(dy) + Math.abs(dz)) * 0.15;
              if (score > bestScore) {
                bestScore = score;
                bestCode = (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1) + 1;
              }
            }
          }
        }
        normalU8[id * 4 + 3] = bestCode;
      }
    }
  }
  return { albedoU8, normalU8, dims };
}

function lookAtMat4(eye, target, up) {
  const fx = target[0] - eye[0], fy = target[1] - eye[1], fz = target[2] - eye[2];
  const fl = Math.hypot(fx, fy, fz) || 1;
  const zx = -fx / fl, zy = -fy / fl, zz = -fz / fl;
  let rx = up[1] * zz - up[2] * zy, ry = up[2] * zx - up[0] * zz, rz = up[0] * zy - up[1] * zx;
  const rl = Math.hypot(rx, ry, rz) || 1;
  rx /= rl; ry /= rl; rz /= rl;
  const ux = zy * rz - zz * ry, uy = zz * rx - zx * rz, uz = zx * ry - zy * rx;
  return new Float32Array([
    rx, ux, zx, 0,
    ry, uy, zy, 0,
    rz, uz, zz, 0,
    -(rx * eye[0] + ry * eye[1] + rz * eye[2]),
    -(ux * eye[0] + uy * eye[1] + uz * eye[2]),
    -(zx * eye[0] + zy * eye[1] + zz * eye[2]),
    1,
  ]);
}

function perspectiveMat4(fovDeg, aspect, near, far) {
  const f = 1 / Math.tan((fovDeg * Math.PI) / 360);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, far * nf, -1,
    0, 0, far * near * nf, 0,
  ]);
}

function mulMat4(a, b) {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] =
        a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
  }
  return out;
}

export class WebGPUFastSDFAO {
  constructor(device, context, format, { width = 1280, height = 720, hasTimestamp = false } = {}) {
    this.device = device;
    this.context = context;
    this.format = format;
    this.width = width;
    this.height = height;
    this.aoW = Math.round(width * 0.5);
    this.aoH = Math.round(height * 0.5);
    this.hasTimestamp = hasTimestamp;
    this.quality = 'high';
    this.useGI = true;
    this.useHash = true;
    this.reuseEnabled = true;
    this.viewMode = 0; // 0=shaded, 1=aoMask, 2=giOnly, 3=hashCache
    this.frame = 0;
    this.sunDir = [0.35, -0.25, 0.90];
    this.emissivePos = [1.2, 1.0, 1.8];
    this.emissiveCol = [0, 0, 0];
    this.emissiveRadius = 0.20;
    this.lastGpuMs = 0;
    this.lastSkipPct = 0;
    this.sceneRes = null;
    this._initPipelines();
    this._initSharedTargets();
  }

  static async create(canvas, opts = {}) {
    if (!navigator.gpu) throw Error('WebGPU is not supported in this browser.');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw Error('No WebGPU adapter available.');
    const hasTimestamp = adapter.features.has('timestamp-query');
    const device = await adapter.requestDevice({
      requiredFeatures: hasTimestamp ? ['timestamp-query'] : [],
    });
    const context = canvas.getContext('webgpu');
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });
    const inst = new WebGPUFastSDFAO(device, context, format, {
      width: canvas.width,
      height: canvas.height,
      hasTimestamp,
      ...opts,
    });
    inst.adapterInfo = adapter.info || {};
    return inst;
  }

  _initPipelines() {
    const d = this.device;
    const vLayout = {
      arrayStride: 36,
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' },
        { shaderLocation: 1, offset: 12, format: 'float32x3' },
        { shaderLocation: 2, offset: 24, format: 'float32x3' },
      ],
    };

    const gMod = d.createShaderModule({ code: GBUFFER_WGSL });
    this.gPipe = d.createRenderPipeline({
      layout: 'auto',
      vertex: { module: gMod, entryPoint: 'vs', buffers: [vLayout] },
      fragment: {
        module: gMod,
        entryPoint: 'fs',
        targets: [{ format: 'rgba32float' }, { format: 'rgba16float' }, { format: 'rgba8unorm' }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });

    const relMod = d.createShaderModule({ code: RELIGHT_WGSL });
    this.relightDirectPipe = d.createComputePipeline({ layout: 'auto', compute: { module: relMod, entryPoint: 'relightDirect' } });
    this.relightBouncePipe = d.createComputePipeline({ layout: 'auto', compute: { module: relMod, entryPoint: 'relightBounce' } });

    const decMod = d.createShaderModule({ code: HASH_DECAY_WGSL });
    this.hashDecayPipe = d.createComputePipeline({ layout: 'auto', compute: { module: decMod, entryPoint: 'main' } });

    const trMod = d.createShaderModule({ code: TRACE_WGSL });
    this.tracePipe = d.createComputePipeline({ layout: 'auto', compute: { module: trMod, entryPoint: 'main' } });

    const dnMod = d.createShaderModule({ code: DENOISE_WGSL });
    this.denoisePipe = d.createComputePipeline({ layout: 'auto', compute: { module: dnMod, entryPoint: 'main' } });

    const cpMod = d.createShaderModule({ code: COMPOSITE_WGSL });
    this.compPipe = d.createRenderPipeline({
      layout: 'auto',
      vertex: { module: cpMod, entryPoint: 'vs', buffers: [vLayout] },
      fragment: { module: cpMod, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });

    this.linSamp = d.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge' });
  }

  _initSharedTargets() {
    const d = this.device;
    const mkTex = (w, h, format, usage) => d.createTexture({ size: [w, h, 1], format, usage });
    const rw = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;
    const rt = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT;

    this.gPosDepth = mkTex(this.aoW, this.aoH, 'rgba32float', rt);
    this.gNormal   = mkTex(this.aoW, this.aoH, 'rgba16float', rt);
    this.gAlbedo   = mkTex(this.aoW, this.aoH, 'rgba8unorm', rt);
    this.gDepth    = mkTex(this.aoW, this.aoH, 'depth24plus', GPUTextureUsage.RENDER_ATTACHMENT);
    this.fullDepth = mkTex(this.width, this.height, 'depth24plus', GPUTextureUsage.RENDER_ATTACHMENT);

    this.rawAOGI   = mkTex(this.aoW, this.aoH, 'rgba16float', rw);
    this.denoiseA  = mkTex(this.aoW, this.aoH, 'rgba16float', rw);
    this.denoiseB  = mkTex(this.aoW, this.aoH, 'rgba16float', rw);
    this.metaTex   = mkTex(this.aoW, this.aoH, 'rgba16float', rw);

    this.hashBuf = d.createBuffer({ size: HASH_SLOTS * 28, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.statsBuf = d.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.statsRead = d.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    this.camUBuf = d.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.relightUBuf = d.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.decayUBuf = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.traceUBuf = d.createBuffer({ size: 192, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.denoiseUBuf1 = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.denoiseUBuf2 = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.compUBuf = d.createBuffer({ size: 144, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    this.radVolA = d.createTexture({ size: RAD_DIMS, dimension: '3d', format: 'rgba8unorm', usage: rw });
    this.radVolB = d.createTexture({ size: RAD_DIMS, dimension: '3d', format: 'rgba8unorm', usage: rw });

    if (this.hasTimestamp) {
      this.tsQuery = d.createQuerySet({ type: 'timestamp', count: 2 });
      this.tsResolve = d.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      this.tsRead = d.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    }
    this._statsMapped = false;
    this._tsMapped = false;
  }

  loadScene({ positions, normals, colors, indices, sdfHalf, sdfDims, sdfMin, sdfSpan, bvh, contactRange = 0.24, useGI = true }) {
    const d = this.device;
    if (this.sceneRes) {
      this.sceneRes.vb.destroy();
      this.sceneRes.ib.destroy();
      this.sceneRes.sdfTex.destroy();
      this.sceneRes.albVol.destroy();
      this.sceneRes.norVol.destroy();
      this.sceneRes.bvhNodes.destroy();
      this.sceneRes.bvhTris.destroy();
      this.sceneRes.bvhGrid.destroy();
    }

    const vCount = positions.length / 3;
    const inter = new Float32Array(vCount * 9);
    for (let i = 0; i < vCount; i++) {
      inter[i * 9]     = positions[i * 3];
      inter[i * 9 + 1] = positions[i * 3 + 1];
      inter[i * 9 + 2] = positions[i * 3 + 2];
      inter[i * 9 + 3] = normals[i * 3];
      inter[i * 9 + 4] = normals[i * 3 + 1];
      inter[i * 9 + 5] = normals[i * 3 + 2];
      inter[i * 9 + 6] = colors[i * 3];
      inter[i * 9 + 7] = colors[i * 3 + 1];
      inter[i * 9 + 8] = colors[i * 3 + 2];
    }
    const vb = d.createBuffer({ size: inter.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(vb, 0, inter);
    const idxArr = indices instanceof Uint32Array ? indices : new Uint32Array(indices);
    const ib = d.createBuffer({ size: Math.ceil(idxArr.byteLength / 4) * 4, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(ib, 0, idxArr);

    // Upload FP16 3D SDF texture
    const [snx, sny, snz] = sdfDims;
    const sdfTex = d.createTexture({ size: [snx, sny, snz], dimension: '3d', format: 'r16float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    d.queue.writeTexture({ texture: sdfTex }, sdfHalf, { bytesPerRow: snx * 2, rowsPerImage: sny }, [snx, sny, snz]);

    // Build static 64x64x40 albedo + normal volumes for GPU relight
    const vox = buildStaticVoxelAttributes(positions, normals, colors, idxArr, sdfMin, sdfSpan, RAD_DIMS);
    const [rx, ry, rz] = RAD_DIMS;
    const albVol = d.createTexture({ size: RAD_DIMS, dimension: '3d', format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const norVol = d.createTexture({ size: RAD_DIMS, dimension: '3d', format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    d.queue.writeTexture({ texture: albVol }, vox.albedoU8, { bytesPerRow: rx * 4, rowsPerImage: ry }, RAD_DIMS);
    d.queue.writeTexture({ texture: norVol }, vox.normalU8, { bytesPerRow: rx * 4, rowsPerImage: ry }, RAD_DIMS);

    // Direct storage buffers for Dual-Child Quantized SAH Contact BVH2
    const nodeData = bvh.uniforms.contactNodes.value.image.data;
    const triData  = bvh.uniforms.contactTriangles.value.image.data;
    const gridData = bvh.uniforms.contactGrid.value.image.data;
    const mkStorage = (arr) => {
      const sz = Math.max(16, Math.ceil(arr.byteLength / 16) * 16);
      const buf = d.createBuffer({ size: sz, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(buf, 0, arr);
      return buf;
    };
    const bvhNodes = mkStorage(nodeData);
    const bvhTris  = mkStorage(triData);
    const bvhGrid  = mkStorage(gridData);

    const gridMin = bvh.uniforms.contactGridMin.value;
    const rootMin = bvh.uniforms.contactRootMin.value;
    const rootSpan = bvh.uniforms.contactRootSpan.value;

    this.sceneRes = {
      vb, ib, indexCount: idxArr.length, sdfTex, albVol, norVol, bvhNodes, bvhTris, bvhGrid,
      sdfMin, sdfSpan, sdfDims, contactRange,
      gridMin: [gridMin.x, gridMin.y, gridMin.z],
      gridSize: bvh.uniforms.contactGridSize.value,
      gridDims: bvh.gridDims,
      nodeCount: bvh.nodes,
      rootMin: [rootMin.x, rootMin.y, rootMin.z],
      rootSpan: [rootSpan.x, rootSpan.y, rootSpan.z],
    };
    this.useGI = useGI;
    this._rebuildBindGroups();
    this.clearHash();
    this.relight(false);
  }

  _rebuildBindGroups() {
    const d = this.device, s = this.sceneRes;
    this.gBG = d.createBindGroup({ layout: this.gPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.camUBuf } }] });
    this.relightDirectBG = d.createBindGroup({
      layout: this.relightDirectPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.relightUBuf } },
        { binding: 1, resource: s.sdfTex.createView() },
        { binding: 2, resource: this.linSamp },
        { binding: 3, resource: s.albVol.createView() },
        { binding: 4, resource: s.norVol.createView() },
        { binding: 6, resource: this.radVolA.createView() },
      ],
    });
    this.relightBounceBG = d.createBindGroup({
      layout: this.relightBouncePipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.relightUBuf } },
        { binding: 1, resource: s.sdfTex.createView() },
        { binding: 2, resource: this.linSamp },
        { binding: 3, resource: s.albVol.createView() },
        { binding: 4, resource: s.norVol.createView() },
        { binding: 5, resource: this.radVolA.createView() },
        { binding: 6, resource: this.radVolB.createView() },
      ],
    });
    this.decayBG = d.createBindGroup({
      layout: this.hashDecayPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.decayUBuf } },
        { binding: 1, resource: { buffer: this.hashBuf } },
      ],
    });
    const makeTraceBG = (radView) => d.createBindGroup({
      layout: this.tracePipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.traceUBuf } },
        { binding: 1, resource: this.gPosDepth.createView() },
        { binding: 2, resource: this.gNormal.createView() },
        { binding: 3, resource: s.sdfTex.createView() },
        { binding: 4, resource: radView },
        { binding: 5, resource: this.linSamp },
        { binding: 6, resource: { buffer: s.bvhNodes } },
        { binding: 7, resource: { buffer: s.bvhTris } },
        { binding: 8, resource: { buffer: s.bvhGrid } },
        { binding: 9, resource: { buffer: this.hashBuf } },
        { binding: 10, resource: this.rawAOGI.createView() },
        { binding: 11, resource: this.metaTex.createView() },
        { binding: 12, resource: { buffer: this.statsBuf } },
      ],
    });
    this.traceBG1B = makeTraceBG(this.radVolA.createView());
    this.traceBG2B = makeTraceBG(this.radVolB.createView());

    const makeDnBG = (ub, src, dst) => d.createBindGroup({
      layout: this.denoisePipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: ub } },
        { binding: 1, resource: src.createView() },
        { binding: 2, resource: this.gPosDepth.createView() },
        { binding: 3, resource: this.gNormal.createView() },
        { binding: 4, resource: this.metaTex.createView() },
        { binding: 5, resource: dst.createView() },
      ],
    });
    this.denoiseBG1 = makeDnBG(this.denoiseUBuf1, this.rawAOGI, this.denoiseA);
    this.denoiseBG2 = makeDnBG(this.denoiseUBuf2, this.denoiseA, this.denoiseB);

    const makeCompBG = (tex) => d.createBindGroup({
      layout: this.compPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.compUBuf } },
        { binding: 1, resource: tex.createView() },
        { binding: 2, resource: this.gPosDepth.createView() },
        { binding: 3, resource: this.gNormal.createView() },
        { binding: 4, resource: this.metaTex.createView() },
      ],
    });
    this.compBGRaw = makeCompBG(this.rawAOGI);
    this.compBGD1  = makeCompBG(this.denoiseA);
    this.compBGD2  = makeCompBG(this.denoiseB);
  }

  clearHash() {
    this.frame = 0;
    const d = this.device;
    d.queue.writeBuffer(this.decayUBuf, 0, new Uint32Array([HASH_SLOTS, 0, 0, 1]));
    const enc = d.createCommandEncoder();
    const p = enc.beginComputePass();
    p.setPipeline(this.hashDecayPipe);
    p.setBindGroup(0, this.decayBG);
    p.dispatchWorkgroups(Math.ceil(HASH_SLOTS / 256));
    p.end();
    d.queue.submit([enc.finish()]);
  }

  relight(softDecayHash = true) {
    if (!this.sceneRes) return;
    const d = this.device, s = this.sceneRes, q = WEBGPU_QUALITY_PRESETS[this.quality];
    const uBuf = new ArrayBuffer(96);
    const f = new Float32Array(uBuf), u = new Uint32Array(uBuf);
    f.set([s.sdfMin[0], s.sdfMin[1], s.sdfMin[2], 0], 0);
    f.set([1 / s.sdfSpan[0], 1 / s.sdfSpan[1], 1 / s.sdfSpan[2], 0], 4);
    f.set([this.sunDir[0], this.sunDir[1], this.sunDir[2], RAD_SCALE], 8);
    f.set([this.emissivePos[0], this.emissivePos[1], this.emissivePos[2], this.emissiveRadius], 12);
    f.set([this.emissiveCol[0], this.emissiveCol[1], this.emissiveCol[2], q.multiBounce ? 0.72 : 0.0], 16);
    u.set([RAD_DIMS[0], RAD_DIMS[1], RAD_DIMS[2], 0], 20);
    d.queue.writeBuffer(this.relightUBuf, 0, uBuf);

    const enc = d.createCommandEncoder();
    const p1 = enc.beginComputePass();
    p1.setPipeline(this.relightDirectPipe);
    p1.setBindGroup(0, this.relightDirectBG);
    p1.dispatchWorkgroups(RAD_DIMS[0] / 4, RAD_DIMS[1] / 4, RAD_DIMS[2] / 4);
    if (q.multiBounce) {
      p1.setPipeline(this.relightBouncePipe);
      p1.setBindGroup(0, this.relightBounceBG);
      p1.dispatchWorkgroups(RAD_DIMS[0] / 4, RAD_DIMS[1] / 4, RAD_DIMS[2] / 4);
    }
    if (softDecayHash) {
      d.queue.writeBuffer(this.decayUBuf, 0, new Uint32Array([HASH_SLOTS, this.frame, 1, 0]));
      p1.setPipeline(this.hashDecayPipe);
      p1.setBindGroup(0, this.decayBG);
      p1.dispatchWorkgroups(Math.ceil(HASH_SLOTS / 256));
    }
    p1.end();
    d.queue.submit([enc.finish()]);
  }

  setQuality(tier) {
    if (!WEBGPU_QUALITY_PRESETS[tier]) return;
    this.quality = tier;
    this.relight(false);
  }

  render(eye, target, { fovDeg = 37, up = [0, 0, 1] } = {}) {
    if (!this.sceneRes) return;
    const d = this.device, s = this.sceneRes, q = WEBGPU_QUALITY_PRESETS[this.quality];
    const view = lookAtMat4(eye, target, up);
    const proj = perspectiveMat4(fovDeg, this.width / this.height, 0.1, 100);
    const vp = mulMat4(proj, view);

    const camData = new Float32Array(32);
    camData.set(vp, 0);
    camData.set(view, 16);
    d.queue.writeBuffer(this.camUBuf, 0, camData);

    const tBuf = new ArrayBuffer(192);
    const tf = new Float32Array(tBuf), tu = new Uint32Array(tBuf);
    tf.set([s.sdfMin[0], s.sdfMin[1], s.sdfMin[2], 0], 0);
    tf.set([1 / s.sdfSpan[0], 1 / s.sdfSpan[1], 1 / s.sdfSpan[2], 0], 4);
    tf.set([s.gridMin[0], s.gridMin[1], s.gridMin[2], s.gridSize], 8);
    tf.set([s.gridDims[0], s.gridDims[1], s.gridDims[2], s.contactRange], 12);
    tf.set([this.sunDir[0], this.sunDir[1], this.sunDir[2], RAD_SCALE], 16);
    tf.set([this.emissivePos[0], this.emissivePos[1], this.emissivePos[2], this.emissiveRadius], 20);
    tf.set([this.emissiveCol[0], this.emissiveCol[1], this.emissiveCol[2], q.giMaxDist], 24);
    tf.set([this.aoW, this.aoH, this.frame, 0.0005], 28);
    tu.set([q.rays, q.maxSamples, this.useGI ? 1 : 0, this.useHash ? 1 : 0], 32);
    tu.set([q.shadowSteps, HASH_MASK, this.reuseEnabled ? 1 : 0, s.nodeCount], 36);
    tf.set([s.rootMin[0], s.rootMin[1], s.rootMin[2], 0], 40);
    tf.set([s.rootSpan[0], s.rootSpan[1], s.rootSpan[2], 0], 44);
    d.queue.writeBuffer(this.traceUBuf, 0, tBuf);

    d.queue.writeBuffer(this.denoiseUBuf1, 0, new Uint32Array([this.aoW, this.aoH, 1, 0]));
    d.queue.writeBuffer(this.denoiseUBuf2, 0, new Uint32Array([this.aoW, this.aoH, 3, 0]));

    const cBuf = new Float32Array(36);
    cBuf.set(vp, 0);
    cBuf.set([eye[0], eye[1], eye[2], 1], 16);
    cBuf.set([this.sunDir[0], this.sunDir[1], this.sunDir[2], this.useGI ? 1 : 0], 20);
    cBuf.set([this.emissivePos[0], this.emissivePos[1], this.emissivePos[2], this.emissiveRadius], 24);
    cBuf.set([this.emissiveCol[0], this.emissiveCol[1], this.emissiveCol[2], this.viewMode], 28);
    cBuf.set([this.width, this.height, this.aoW, this.aoH], 32);
    d.queue.writeBuffer(this.compUBuf, 0, cBuf);

    d.queue.writeBuffer(this.statsBuf, 0, new Uint32Array([0, 0, 0, 0]));

    const enc = d.createCommandEncoder();

    // Pass 1: Half-Res G-Buffer
    const gp = enc.beginRenderPass({
      colorAttachments: [
        { view: this.gPosDepth.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
        { view: this.gNormal.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
        { view: this.gAlbedo.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
      ],
      depthStencilAttachment: { view: this.gDepth.createView(), depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store' },
      ...(this.hasTimestamp ? { timestampWrites: { querySet: this.tsQuery, beginningOfPassWriteIndex: 0 } } : {}),
    });
    gp.setPipeline(this.gPipe);
    gp.setBindGroup(0, this.gBG);
    gp.setVertexBuffer(0, s.vb);
    gp.setIndexBuffer(s.ib, 'uint32');
    gp.drawIndexed(s.indexCount);
    gp.end();

    // Pass 2 & 3: Trace (BVH2 + 2x2 Occlusion-Point Reuse + SHaRC Hash) + Denoise
    const cp = enc.beginComputePass();
    cp.setPipeline(this.tracePipe);
    cp.setBindGroup(0, q.multiBounce ? this.traceBG2B : this.traceBG1B);
    cp.dispatchWorkgroups(Math.ceil(this.aoW / 8), Math.ceil(this.aoH / 8));
    if (this.useGI && q.denoisePasses >= 1) {
      cp.setPipeline(this.denoisePipe);
      cp.setBindGroup(0, this.denoiseBG1);
      cp.dispatchWorkgroups(Math.ceil(this.aoW / 8), Math.ceil(this.aoH / 8));
      if (q.denoisePasses >= 2) {
        cp.setBindGroup(0, this.denoiseBG2);
        cp.dispatchWorkgroups(Math.ceil(this.aoW / 8), Math.ceil(this.aoH / 8));
      }
    }
    cp.end();

    // Pass 4: Full-Res Composite
    const fp = enc.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0.07, g: 0.09, b: 0.13, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: { view: this.fullDepth.createView(), depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store' },
      ...(this.hasTimestamp ? { timestampWrites: { querySet: this.tsQuery, endOfPassWriteIndex: 1 } } : {}),
    });
    fp.setPipeline(this.compPipe);
    fp.setBindGroup(0, !this.useGI ? this.compBGRaw : (q.denoisePasses >= 2 ? this.compBGD2 : this.compBGD1));
    fp.setVertexBuffer(0, s.vb);
    fp.setIndexBuffer(s.ib, 'uint32');
    fp.drawIndexed(s.indexCount);
    fp.end();

    if (!this._statsMapped) {
      enc.copyBufferToBuffer(this.statsBuf, 0, this.statsRead, 0, 16);
    }
    if (this.hasTimestamp && !this._tsMapped) {
      enc.resolveQuerySet(this.tsQuery, 0, 2, this.tsResolve, 0);
      enc.copyBufferToBuffer(this.tsResolve, 0, this.tsRead, 0, 16);
    }
    d.queue.submit([enc.finish()]);
    this.frame++;

    if (!this._statsMapped) {
      this._statsMapped = true;
      this.statsRead.mapAsync(GPUMapMode.READ).then(() => {
        const arr = new Uint32Array(this.statsRead.getMappedRange());
        const tr = arr[0], sk = arr[1];
        this.lastSkipPct = (tr + sk) > 0 ? (100 * sk) / (tr + sk) : 0;
        this.statsRead.unmap();
        this._statsMapped = false;
      }).catch(() => { this._statsMapped = false; });
    }
    if (this.hasTimestamp && !this._tsMapped) {
      this._tsMapped = true;
      this.tsRead.mapAsync(GPUMapMode.READ).then(() => {
        const ts = new BigInt64Array(this.tsRead.getMappedRange());
        const dt = Number(ts[1] - ts[0]) / 1e6;
        if (dt > 0 && dt < 200) this.lastGpuMs = dt;
        this.tsRead.unmap();
        this._tsMapped = false;
      }).catch(() => { this._tsMapped = false; });
    }
  }
}
