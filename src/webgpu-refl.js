// Real-Time Glossy / Rough SDF Specular Reflections for WebGPU Compute:
// - 1 GGX-importance-sampled reflection ray per half-res pixel (roughness < 0.72)
// - Hybrid trace: Storage-buffer Dual-Child SAH BVH2 (near contact) -> FP16 SDF (far) -> radVol + direct Emissive Orb hit
// - Temporal reprojection EMA + roughness-guided 3x3 bilateral spatial filter

export const REFL_TRACE_WGSL = /* wgsl */`
struct ReflU {
  sdfMin: vec4<f32>,
  sdfInvSpan: vec4<f32>,
  gridMinSize: vec4<f32>,   // xyz=gridMin, w=gridSize
  gridDimsRange: vec4<f32>, // xyz=gridDims, w=contactRange
  eyeFrame: vec4<f32>,      // xyz=eye, w=frame
  emissivePos: vec4<f32>,   // xyz=pos, w=radius
  emissiveCol: vec4<f32>,   // rgb=col, w=radScale
  params: vec4<f32>,        // x=W, y=H, z=contactBias, w=maxDist
  rootMin: vec4<f32>,
  rootSpan: vec4<f32>,
  extra: vec4<u32>,         // x=nodeCount, y=enabled(0/1), z=0, w=0
};

@group(0) @binding(0) var<uniform> u: ReflU;
@group(0) @binding(1) var gPosDepth: texture_2d<f32>;
@group(0) @binding(2) var gNormal: texture_2d<f32>;
@group(0) @binding(3) var gAlbedoRough: texture_2d<f32>;
@group(0) @binding(4) var sdfTex: texture_3d<f32>;
@group(0) @binding(5) var radVol: texture_3d<f32>;
@group(0) @binding(6) var linSamp: sampler;
@group(0) @binding(7) var<storage, read> bvhNodes: array<vec4<u32>>;
@group(0) @binding(8) var<storage, read> bvhTris: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read> bvhGrid: array<vec2<u32>>;
@group(0) @binding(10) var outRefl: texture_storage_2d<rgba16float, write>;

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

fn sampleSDF(p: vec3<f32>) -> f32 {
  let uv = (p - u.sdfMin.xyz) * u.sdfInvSpan.xyz;
  if (any(uv < vec3<f32>(0.0)) || any(uv > vec3<f32>(1.0))) { return 1e4; }
  return textureSampleLevel(sdfTex, linSamp, uv, 0.0).r;
}

fn sampleRad(p: vec3<f32>) -> vec3<f32> {
  let uv = (p - u.sdfMin.xyz) * u.sdfInvSpan.xyz;
  if (any(uv < vec3<f32>(0.0)) || any(uv > vec3<f32>(1.0))) { return vec3<f32>(0.0); }
  return textureSampleLevel(radVol, linSamp, uv, 0.0).rgb * u.emissiveCol.w;
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

fn testLeafHit(leafMeta: u32, origin: vec3<f32>, w: vec3<f32>, bestT: ptr<function, f32>) -> bool {
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
      hit = true;
    }
  }
  return hit;
}

fn bvhContactHit(origin: vec3<f32>, w: vec3<f32>, root: u32, cnt: u32, qMin: vec3<f32>, qScale: vec3<f32>, bestT: ptr<function, f32>) -> bool {
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
    return testLeafHit(nd.w, origin, w, bestT);
  }
  var anyHit = false;
  var stack: array<u32, 16>;
  var sp = 0u;
  var pair = root;
  for (var visit = 0u; visit < 256u; visit++) {
    let ndL = bvhNodes[pair];
    let ndR = bvhNodes[pair + 1u];
    let tL = boxHit(ndL, qInv, qOff, *bestT);
    let tR = boxHit(ndR, qInv, qOff, *bestT);
    var hitL = tL >= 0.0 && tL < *bestT;
    var hitR = tR >= 0.0 && tR < *bestT;
    if (hitL && (ndL.w & 0x80000000u) != 0u) {
      if (testLeafHit(ndL.w, origin, w, bestT)) { anyHit = true; }
      hitL = false;
    }
    if (hitR && (ndR.w & 0x80000000u) != 0u) {
      if (testLeafHit(ndR.w, origin, w, bestT)) { anyHit = true; }
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

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let W = u32(u.params.x);
  let H = u32(u.params.y);
  if (gid.x >= W || gid.y >= H) { return; }
  let q = vec2<i32>(gid.xy);

  if (u.extra.y == 0u) {
    textureStore(outRefl, q, vec4<f32>(0.0));
    return;
  }

  let pd = textureLoad(gPosDepth, q, 0);
  let gn = textureLoad(gNormal, q, 0);
  let ar = textureLoad(gAlbedoRough, q, 0);
  let rough = clamp(ar.a, 0.05, 1.0);
  if (pd.w <= 0.0 || gn.w < 0.5 || rough > 0.72) {
    textureStore(outRefl, q, vec4<f32>(0.0, 0.0, 0.0, 1.0));
    return;
  }

  let p = pd.xyz;
  let n = normalize(gn.xyz);
  let V = normalize(u.eyeFrame.xyz - p);
  let frame = u32(u.eyeFrame.w);
  let px = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);

  // GGX importance-sampled half vector (clamped alpha for crisp low-variance reflections)
  let a = rough * rough;
  let u1 = clamp(fract(ign(px) + vdc2(frame)), 1e-4, 0.92);
  let u2 = fract(ign(px + vec2<f32>(29.0, 53.0)) + vdc3(frame));
  let phi = 6.2831853 * u2;
  let cosTh = sqrt((1.0 - u1) / max(1.0 + (a * a - 1.0) * u1, 1e-5));
  let sinTh = sqrt(max(0.0, 1.0 - cosTh * cosTh));

  let ax = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.x) < 0.9);
  let tVec = normalize(cross(n, ax));
  let bVec = cross(n, tVec);
  let Hvec = normalize(tVec * (sinTh * cos(phi)) + bVec * (sinTh * sin(phi)) + n * cosTh);
  var R = reflect(-V, Hvec);
  if (dot(R, n) < 0.03) {
    R = normalize(reflect(-V, n) + n * 0.05);
  }

  let origin = p + n * 0.008;
  let cRange = u.gridDimsRange.w;
  var hitDist = u.params.w;
  var rad = vec3<f32>(0.0);
  var hitAny = false;

  // Check direct emissive orb sphere intersection first!
  var orbT = 1e9;
  if (dot(u.emissiveCol.rgb, u.emissiveCol.rgb) > 1e-4) {
    let oc = origin - u.emissivePos.xyz;
    let b = dot(oc, R);
    let c = dot(oc, oc) - u.emissivePos.w * u.emissivePos.w;
    let disc = b * b - c;
    if (disc >= 0.0) {
      let t0 = -b - sqrt(disc);
      if (t0 > 0.01 && t0 < hitDist) { orbT = t0; }
    }
  }

  // 1) Near-contact exact SAH BVH2 reflection
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
      cellQMin = u.gridMinSize.xyz + vec3<f32>(cell) * gSize - vec3<f32>(cRange + 0.002);
      cellQScale = vec3<f32>((gSize + 2.0 * (cRange + 0.002)) * (1.0 / 65535.0));
    }
  } else {
    cellCnt = u.extra.x;
  }

  var tBvh = cRange;
  if (bvhContactHit(origin, R, cellRoot, cellCnt, cellQMin, cellQScale, &tBvh) && tBvh < orbT) {
    hitAny = true;
    hitDist = tBvh;
    rad = sampleRad(origin + R * tBvh);
  } else {
    // 2) Far SDF sphere trace
    var s = cRange;
    for (var k = 0u; k < 28u; k++) {
      if (s >= orbT) { break; }
      let d = sampleSDF(origin + R * s);
      if (d > 100.0) { break; }
      if (d < 0.025) {
        hitAny = true;
        hitDist = s;
        rad = sampleRad(origin + R * s);
        break;
      }
      s += max(d * 0.85, 0.035);
      if (s > u.params.w) { break; }
    }
  }

  if (orbT < hitDist) {
    hitAny = true;
    hitDist = orbT;
    rad = u.emissiveCol.rgb * 1.8;
  }
  if (!hitAny) {
    // Clean studio sky gradient fallback on open ceiling / exterior miss
    let skyT = clamp(R.z * 0.5 + 0.5, 0.0, 1.0);
    rad = mix(vec3<f32>(0.18, 0.20, 0.24), vec3<f32>(0.55, 0.66, 0.82), skyT);
  }

  let normDist = clamp(hitDist / 6.0, 0.05, 1.0);
  textureStore(outRefl, q, vec4<f32>(rad, normDist));
}
`;

export const REFL_FILTER_WGSL = /* wgsl */`
struct ReflFiltU {
  prevViewProj: mat4x4<f32>,
  params: vec4<f32>, // x=W, y=H, z=alpha, w=validPrev(0/1)
};

@group(0) @binding(0) var<uniform> u: ReflFiltU;
@group(0) @binding(1) var curRefl: texture_2d<f32>;
@group(0) @binding(2) var curPosDepth: texture_2d<f32>;
@group(0) @binding(3) var curNormal: texture_2d<f32>;
@group(0) @binding(4) var gAlbedoRough: texture_2d<f32>;
@group(0) @binding(5) var prevRefl: texture_2d<f32>;
@group(0) @binding(6) var prevPosDepth: texture_2d<f32>;
@group(0) @binding(7) var prevNormal: texture_2d<f32>;
@group(0) @binding(8) var outRefl: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let W = i32(u.params.x);
  let H = i32(u.params.y);
  let q = vec2<i32>(gid.xy);
  if (q.x >= W || q.y >= H) { return; }

  let pd = textureLoad(curPosDepth, q, 0);
  let gn = textureLoad(curNormal, q, 0);
  let ar = textureLoad(gAlbedoRough, q, 0);
  let rough = clamp(ar.a, 0.05, 1.0);
  let c0 = textureLoad(curRefl, q, 0);
  if (pd.w <= 0.0 || gn.w < 0.5 || rough > 0.72) {
    textureStore(outRefl, q, c0);
    return;
  }

  let p = pd.xyz;
  let n = normalize(gn.xyz);
  let planeTol = 0.0025 * pd.w + 0.012;

  // Roughness-guided 3x3 spatial pre-filter (tight on mirror-like surfaces, wider on rough surfaces)
  var spSum = c0;
  var spW = 1.0;
  let spGain = smoothstep(0.08, 0.35, rough);
  if (spGain > 0.02) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        if (dx == 0 && dy == 0) { continue; }
        let nq = q + vec2<i32>(dx, dy);
        if (nq.x < 0 || nq.y < 0 || nq.x >= W || nq.y >= H) { continue; }
        let npd = textureLoad(curPosDepth, nq, 0);
        let ngn = textureLoad(curNormal, nq, 0);
        if (npd.w <= 0.0 || ngn.w < 0.5) { continue; }
        if (dot(normalize(ngn.xyz), n) < 0.92) { continue; }
        if (abs(dot(n, npd.xyz - p)) > planeTol) { continue; }
        let w = select(0.25, 0.5, dx == 0 || dy == 0) * spGain;
        spSum += textureLoad(curRefl, nq, 0) * w;
        spW += w;
      }
    }
  }
  let filtered = spSum / spW;

  if (u.params.w < 0.5) {
    textureStore(outRefl, q, filtered);
    return;
  }

  // Temporal reprojection EMA
  let clip = u.prevViewProj * vec4<f32>(p, 1.0);
  if (clip.w <= 0.0) {
    textureStore(outRefl, q, filtered);
    return;
  }
  let ndc = clip.xy / clip.w;
  let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  var f = uv * vec2<f32>(u.params.xy) - 0.5;
  f = select(f, round(f), abs(f - round(f)) < vec2<f32>(0.02));
  let i0 = vec2<i32>(floor(f));
  let fw = f - floor(f);

  var acc = vec4<f32>(0.0);
  var wSum = 0.0;
  for (var k = 0; k < 4; k++) {
    let o = vec2<i32>(k & 1, k >> 1);
    let pq = i0 + o;
    let bw = select(1.0 - fw.x, fw.x, o.x == 1) * select(1.0 - fw.y, fw.y, o.y == 1);
    if (bw < 1e-4 || pq.x < 0 || pq.y < 0 || pq.x >= W || pq.y >= H) { continue; }
    let ppd = textureLoad(prevPosDepth, pq, 0);
    let pgn = textureLoad(prevNormal, pq, 0);
    if (ppd.w <= 0.0 || pgn.w < 0.5) { continue; }
    if (dot(normalize(pgn.xyz), n) < 0.92) { continue; }
    if (abs(dot(n, ppd.xyz - p)) > planeTol) { continue; }
    acc += textureLoad(prevRefl, pq, 0) * bw;
    wSum += bw;
  }
  if (wSum > 0.01) {
    textureStore(outRefl, q, mix(acc / wSum, filtered, u.params.z));
  } else {
    textureStore(outRefl, q, filtered);
  }
}
`;
