// Fast-SDF-AO: Drop-in Three.js WebGL2 API for real-time signed distance field AO.
// Exports:
//   - loadSDF(urlOrBuffer, { dims, format, gzip, THREE })
//   - GENERIC_SDF_AO_GLSL
//   - patchMaterialWithSDFAO(material, { field, min, span, radius, intensity, multiBounce })
//   - TracedSDFAO
//   - buildContactBVH, CONTACT_GLSL (re-exported from ../contact-bvh.js)

import { buildContactBVH, CONTACT_GLSL } from '../contact-bvh.js';

export { buildContactBVH, CONTACT_GLSL };

/**
 * Scene-agnostic 4-tap SDF AO GLSL (no wallFace metadata) + Jimenez 2016
 * multi-bounce colored AO + Lagarde/Jimenez GTSO specular occlusion.
 */
export const GENERIC_SDF_AO_GLSL = /* glsl */ `
precision highp sampler3D;
uniform sampler3D uSdfField;
uniform vec3 uSdfMin;
uniform vec3 uSdfInvSpan;
uniform float uSdfRadius;

float sdfSampleStatic(vec3 pos) {
  vec3 uv = (pos - uSdfMin) * uSdfInvSpan;
  if (any(lessThan(uv, vec3(0.0))) || any(greaterThan(uv, vec3(1.0)))) return 1e4;
  return texture(uSdfField, uv).r;
}

// Generic 4-tap SDF AO along surface normal (no wall metadata required).
float sdfAOFourTap(vec3 pos, vec3 normal) {
  vec4 r = vec4(0.142857, 0.428571, 0.714286, 1.0) * uSdfRadius;
  vec4 d = vec4(
    sdfSampleStatic(pos + normal * r.x),
    sdfSampleStatic(pos + normal * r.y),
    sdfSampleStatic(pos + normal * r.z),
    sdfSampleStatic(pos + normal * r.w)
  );
  vec4 w = vec4(1.0, 0.816667, 0.633333, 0.45);
  vec4 occ = clamp((r - d - 0.045) / r, 0.0, 1.0);
  return clamp(1.0 - 0.92 * dot(w, occ) / 2.9, 0.0, 1.0);
}

// Jimenez 2016 Multi-Bounce Colored AO approximation.
vec3 multiBounceAO(float ao, vec3 albedo) {
  vec3 a = 2.0404 * albedo - 0.3324;
  vec3 b = -4.7951 * albedo + 0.6417;
  vec3 c = 2.7552 * albedo + 0.6903;
  return max(vec3(ao), ((ao * a + b) * ao + c) * ao);
}

// Jimenez / Lagarde GTSO Specular Occlusion from visibility cone + GGX roughness.
float specularOcclusion(float NoV, float ao, float roughness) {
  return clamp(pow(NoV + ao, exp2(-16.0 * roughness - 1.0)) - 1.0 + ao, 0.0, 1.0);
}
`;

async function decompressGzip(bytes) {
  if (typeof DecompressionStream === 'undefined') {
    throw new Error('DecompressionStream is unavailable in this environment');
  }
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).arrayBuffer();
}

function toVec3Array(v, name) {
  if (Array.isArray(v) && v.length === 3) return [v[0], v[1], v[2]];
  if (v && typeof v.x === 'number' && typeof v.y === 'number' && typeof v.z === 'number') {
    return [v.x, v.y, v.z];
  }
  throw new Error(`Invalid ${name}: expected [x, y, z] or THREE.Vector3`);
}

/**
 * Load a baked .f16, .f16.gz, .f32, or .f32.gz signed distance field into a
 * configured THREE.Data3DTexture (RedFormat, HalfFloatType/FloatType, LinearFilter).
 */
export async function loadSDF(urlOrArrayBuffer, { dims, format = 'f16', gzip, THREE } = {}) {
  if (!Array.isArray(dims) || dims.length !== 3) {
    throw new Error('loadSDF requires { dims: [nx, ny, nz] }');
  }
  if (format !== 'f16' && format !== 'f32') {
    throw new Error(`Unsupported SDF format "${format}" (expected "f16" or "f32")`);
  }

  let rawBuffer;
  let isGzip = gzip;
  if (typeof urlOrArrayBuffer === 'string') {
    const resp = await fetch(urlOrArrayBuffer);
    if (!resp.ok) throw new Error(`Failed to fetch SDF "${urlOrArrayBuffer}" (${resp.status})`);
    if (isGzip === undefined) isGzip = urlOrArrayBuffer.endsWith('.gz');
    rawBuffer = await resp.arrayBuffer();
  } else if (urlOrArrayBuffer instanceof ArrayBuffer) {
    rawBuffer = urlOrArrayBuffer;
  } else if (ArrayBuffer.isView(urlOrArrayBuffer)) {
    rawBuffer = urlOrArrayBuffer.buffer.slice(
      urlOrArrayBuffer.byteOffset,
      urlOrArrayBuffer.byteOffset + urlOrArrayBuffer.byteLength
    );
  } else {
    throw new Error('loadSDF expects a URL string, ArrayBuffer, or TypedArray');
  }

  const bytes = new Uint8Array(rawBuffer);
  if (isGzip === undefined) {
    isGzip = bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;
  }
  const buffer = isGzip ? await decompressGzip(bytes) : rawBuffer;

  const [nx, ny, nz] = dims;
  const expectedVoxels = nx * ny * nz;
  const data = format === 'f16' ? new Uint16Array(buffer) : new Float32Array(buffer);
  if (data.length !== expectedVoxels) {
    throw new Error(
      `SDF voxel count mismatch: got ${data.length}, expected ${expectedVoxels} (${nx}x${ny}x${nz})`
    );
  }

  const T = THREE || (await import('three'));
  const tex = new T.Data3DTexture(data, nx, ny, nz);
  tex.format = T.RedFormat;
  tex.type = format === 'f16' ? T.HalfFloatType : T.FloatType;
  tex.minFilter = T.LinearFilter;
  tex.magFilter = T.LinearFilter;
  tex.generateMipmaps = false;
  tex.unpackAlignment = 1;
  tex.needsUpdate = true;
  return tex;
}

/**
 * Drop inline 4-tap SDF AO (+ optional Jimenez multi-bounce & GTSO specular occlusion)
 * directly into any standard Three.js material (MeshStandardMaterial, MeshPhysicalMaterial,
 * MeshLambertMaterial) via onBeforeCompile with zero extra render passes.
 */
export function patchMaterialWithSDFAO(
  material,
  { field, min, span, radius = 1.12, intensity = 1.0, multiBounce = true } = {}
) {
  if (!material) throw new Error('patchMaterialWithSDFAO requires a Three.js material');
  if (!field) throw new Error('patchMaterialWithSDFAO requires a Data3DTexture { field }');
  const minArr = toVec3Array(min, 'min');
  const spanArr = toVec3Array(span, 'span');

  const sdfUniforms = {
    uSdfField: { value: field },
    uSdfMin: { value: new Float32Array(minArr) },
    uSdfInvSpan: { value: new Float32Array([1 / spanArr[0], 1 / spanArr[1], 1 / spanArr[2]]) },
    uSdfRadius: { value: radius },
    uSdfAoIntensity: { value: intensity },
    uSdfAoMultiBounce: { value: multiBounce ? 1.0 : 0.0 },
  };

  const prevCompile = material.onBeforeCompile;
  material.onBeforeCompile = (shader, renderer) => {
    if (prevCompile) prevCompile.call(material, shader, renderer);
    Object.assign(shader.uniforms, sdfUniforms);

    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <common>',
        `#include <common>
varying vec3 vSdfWorldPos;
varying vec3 vSdfWorldNormal;`
      )
      .replace(
        '#include <worldpos_vertex>',
        `#include <worldpos_vertex>
vec4 _sdfWp = vec4( transformed, 1.0 );
vec3 _sdfWn = objectNormal;
#ifdef USE_INSTANCING
  _sdfWp = instanceMatrix * _sdfWp;
  _sdfWn = mat3( instanceMatrix ) * _sdfWn;
#endif
_sdfWp = modelMatrix * _sdfWp;
vSdfWorldPos = _sdfWp.xyz;
vSdfWorldNormal = normalize( mat3( modelMatrix ) * _sdfWn );`
      );

    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        `#include <common>
varying vec3 vSdfWorldPos;
varying vec3 vSdfWorldNormal;
uniform float uSdfAoIntensity;
uniform float uSdfAoMultiBounce;
${GENERIC_SDF_AO_GLSL}`
      )
      .replace(
        '#include <aomap_fragment>',
        `#include <aomap_fragment>
{
  float _sdfAo = clamp( mix( 1.0, sdfAOFourTap( vSdfWorldPos, normalize( vSdfWorldNormal ) ), uSdfAoIntensity ), 0.0, 1.0 );
  vec3 _sdfDiffAo = uSdfAoMultiBounce > 0.5 ? multiBounceAO( _sdfAo, diffuseColor.rgb ) : vec3( _sdfAo );
  reflectedLight.indirectDiffuse *= _sdfDiffAo;
  #ifdef STANDARD
    float _sdfNoV = clamp( dot( geometryNormal, geometryViewDir ), 0.0, 1.0 );
    reflectedLight.indirectSpecular *= specularOcclusion( _sdfNoV, _sdfAo, material.roughness );
  #endif
}`
      );
  };

  const prevKey = material.customProgramCacheKey?.bind(material);
  material.customProgramCacheKey = () => (prevKey ? prevKey() + '|' : '') + 'fast-sdf-ao-v1';
  material.userData.sdfAO = sdfUniforms;
  material.needsUpdate = true;
  return sdfUniforms;
}

const GBUFFER_VERT = /* glsl */ `
out vec3 vPos;
out vec3 vNormal;
void main() {
  mat4 wm = modelMatrix;
  #ifdef USE_INSTANCING
    wm = wm * instanceMatrix;
  #endif
  vec4 wp = wm * vec4(position, 1.0);
  vPos = wp.xyz;
  vNormal = normalize(mat3(wm) * normal);
  gl_Position = projectionMatrix * viewMatrix * wp;
}
`;

const GBUFFER_FRAG = /* glsl */ `
precision highp float;
in vec3 vPos;
in vec3 vNormal;
layout(location = 0) out vec4 gPosOut;
layout(location = 1) out vec4 gNormalOut;
void main() {
  gPosOut = vec4(vPos, -(viewMatrix * vec4(vPos, 1.0)).z);
  gNormalOut = vec4(normalize(vNormal), 0.0);
}
`;

const TRACED_AO_FRAG = /* glsl */ `
precision highp float;
precision highp sampler3D;
uniform sampler3D field;
uniform vec3 sdfMin;
uniform vec3 sdfInvSpan;
uniform sampler2D gPos;
uniform sampler2D gNormal;
uniform sampler2D histTex;
uniform sampler2D normalTex;
#ifdef BENT_NORMALS
uniform sampler2D bentTex;
#endif
uniform mat4 prevViewProj;
uniform mat4 prevView;
uniform vec2 invProjXY;
uniform float frame;
uniform float histMax;
uniform float settleFrames;
layout(location = 0) out vec4 histOut;
layout(location = 1) out vec4 normalOut;
#ifdef BENT_NORMALS
layout(location = 2) out vec4 bentOut;
#endif

vec2 octEncode(vec3 n) {
  n /= abs(n.x) + abs(n.y) + abs(n.z);
  if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
  return n.xy * 0.5 + 0.5;
}
vec3 octDecodeRaw(vec2 e) {
  e = e * 2.0 - 1.0;
  vec3 v = vec3(e.xy, 1.0 - abs(e.x) - abs(e.y));
  if (v.z < 0.0) v.xy = (1.0 - abs(v.yx)) * vec2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
  return v;
}
float staticDistance(vec3 pos) {
  vec3 uv = (pos - sdfMin) * sdfInvSpan;
  if (any(lessThan(uv, vec3(0.0))) || any(greaterThan(uv, vec3(1.0)))) return 1e4;
  return texture(field, uv).r;
}
float ign(vec2 q) { return fract(52.9829189 * fract(dot(q, vec2(0.06711056, 0.00583715)))); }
${CONTACT_GLSL.replaceAll('candidateAO(', 'tracedAO(').replaceAll('candidateAOBent(', 'tracedAOBent(')}

vec3 history(vec3 p, vec3 n, out vec3 prevBent) {
  prevBent = n;
  vec4 c = prevViewProj * vec4(p, 1.0);
  if (c.w <= 0.0) return vec3(0.0);
  vec2 res = vec2(textureSize(histTex, 0));
  vec2 invRes2 = 2.0 / res;
  vec2 f = (c.xy / c.w * 0.5 + 0.5) * res - 0.5;
  f = mix(f, round(f), vec2(lessThan(abs(f - round(f)), vec2(0.02))));
  ivec2 i0 = ivec2(floor(f));
  vec2 w = f - floor(f);
  vec3 acc = vec3(0.0);
  float ws = 0.0;
  #ifdef BENT_NORMALS
  vec3 bAcc = vec3(0.0);
  #endif
  vec3 vp = (prevView * vec4(p, 1.0)).xyz;
  vec3 vn = mat3(prevView) * n;
  vec2 vnXY = vn.xy * invProjXY;
  float vpDotN = dot(vn, vp);
  float planeTol = 0.0015 * c.w + 0.005;
  float depthTol = 0.08 * c.w + 0.05;
  for (int k = 0; k < 4; k++) {
    ivec2 o = ivec2(k & 1, k >> 1), q = i0 + o;
    float bw = (o.x == 1 ? w.x : 1.0 - w.x) * (o.y == 1 ? w.y : 1.0 - w.y);
    if (bw < 1e-4 || any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, ivec2(res)))) continue;
    vec2 h = texelFetch(histTex, q, 0).xy;
    vec4 hn = texelFetch(normalTex, q, 0);
    float frames = hn.w * 255.0;
    if (frames < 0.5 || abs(h.y - c.w) > depthTol) continue;
    vec3 nv = octDecodeRaw(hn.xy);
    float dn = dot(nv, n);
    if (dn <= 0.0 || dn * dn < 0.81 * dot(nv, nv)) continue;
    vec2 ndcQ = (vec2(q) + 0.5) * invRes2 - 1.0;
    if (abs((dot(vnXY, ndcQ) - vn.z) * h.y - vpDotN) > planeTol) continue;
    float sig = hn.z * 0.25;
    acc += bw * vec3(h.x, frames, sig * sig);
    #ifdef BENT_NORMALS
    bAcc += bw * (texelFetch(bentTex, q, 0).xyz * 2.0 - 1.0);
    #endif
    ws += bw;
  }
  if (ws > 0.01) {
    #ifdef BENT_NORMALS
    float bl = length(bAcc);
    prevBent = bl > 1e-4 ? bAcc / bl : n;
    #endif
    return acc / ws;
  }
  return vec3(0.0);
}

void main() {
  ivec2 q = ivec2(gl_FragCoord.xy);
  vec4 g = texelFetch(gPos, q, 0);
  vec4 gn = texelFetch(gNormal, q, 0);
  if (g.w <= 0.0) {
    histOut = vec4(0.0, g.w, 0.0, 1.0);
    normalOut = vec4(0.0);
    #ifdef BENT_NORMALS
    bentOut = vec4(0.5, 0.5, 1.0, 1.0);
    #endif
    return;
  }
  vec3 p = g.xyz, normal = normalize(gn.xyz);
  vec3 prevBent;
  vec3 h = history(p, normal, prevBent);
  bool settled = settleFrames > 0.0 && (h.y >= settleFrames || (h.y >= 24.0 && h.z < 0.00015));
  float aoStatic, aoVar, frames;
  #ifdef BENT_NORMALS
  vec3 bentN;
  #endif
  if (!settled) {
    frames = min(h.y + 1.0, histMax);
    float nd = h.y < 8.0 ? 4.0 : (h.y < 16.0 ? 3.0 : (h.y >= 24.0 && h.z < 0.0006 ? 1.0 : 2.0));
    vec4 resAO = tracedAOBent(p, normal, nd);
    float sampleAO = resAO.x;
    float alpha = 1.0 / frames;
    aoStatic = mix(h.x, sampleAO, alpha);
    aoVar = frames > 1.5 ? max(0.0, ((frames - 1.0) * h.z + (sampleAO - h.x) * (sampleAO - aoStatic)) / frames) : 0.02;
    #ifdef BENT_NORMALS
    vec3 bMix = mix(prevBent, resAO.yzw, alpha);
    float bl = length(bMix);
    bentN = bl > 1e-4 ? bMix / bl : normal;
    #endif
  } else {
    frames = min(h.y, histMax);
    aoStatic = h.x;
    aoVar = h.z;
    #ifdef BENT_NORMALS
    bentN = prevBent;
    #endif
  }
  histOut = vec4(aoStatic, g.w, 0.0, 1.0);
  normalOut = vec4(octEncode(normal), clamp(sqrt(aoVar) * 4.0, 0.0, 1.0), frames / 255.0);
  #ifdef BENT_NORMALS
  bentOut = vec4(bentN * 0.5 + 0.5, 1.0);
  #endif
}
`;

/**
 * Reusable Traced SDF + Quantized SAH Contact BVH renderer for Three.js WebGL2.
 * Manages the half-res geometry pass, quantized SAH contact BVH, RG32F+RGBA8
 * Welford variance temporal history, and optional bent-normal MRT.
 */
export class TracedSDFAO {
  constructor(
    renderer,
    scene,
    camera,
    {
      THREE,
      field,
      min,
      span,
      staticMeshes = [],
      width = 1280,
      height = 720,
      aoScale = 0.5,
      gridSize = 1.0,
      bias = 0.0003,
      bentNormals = false,
      smoothFalloff = false,
      settleFrames = 64,
    } = {}
  ) {
    if (!THREE) throw new Error('TracedSDFAO requires { THREE } (or use TracedSDFAO.create(...))');
    if (!field) throw new Error('TracedSDFAO requires a Data3DTexture { field }');
    this.T = THREE;
    this.renderer = renderer;
    this.scene = scene;
    this.camera = camera;
    this.width = width;
    this.height = height;
    this.aoScale = aoScale;
    this.bentNormals = bentNormals;
    this.histIndex = 0;
    this.frame = 0;

    const minArr = toVec3Array(min, 'min');
    const spanArr = toVec3Array(span, 'span');
    this.min = new THREE.Vector3(...minArr);
    this.span = new THREE.Vector3(...spanArr);
    this.invSpan = new THREE.Vector3(1 / spanArr[0], 1 / spanArr[1], 1 / spanArr[2]);
    this.field = field;
    this.prevViewProj = new THREE.Matrix4();
    this.prevView = new THREE.Matrix4();
    this.curViewProj = new THREE.Matrix4();
    this.curView = new THREE.Matrix4();
    this.lastCamera = new THREE.Matrix4();
    this.invProjXY = new THREE.Vector2(1, 1);

    const w = Math.max(1, Math.round(width * aoScale));
    const h = Math.max(1, Math.round(height * aoScale));
    const nearest = (t) => {
      t.minFilter = THREE.NearestFilter;
      t.magFilter = THREE.NearestFilter;
      t.generateMipmaps = false;
    };

    this.gTarget = new THREE.WebGLRenderTarget(w, h, { count: 2 });
    this.gTarget.textures[0].type = THREE.FloatType;
    this.gTarget.textures[1].type = THREE.HalfFloatType;
    this.gTarget.textures.forEach(nearest);

    const mrtCount = bentNormals ? 3 : 2;
    this.histories = [0, 1].map(() => {
      const rt = new THREE.WebGLRenderTarget(w, h, { count: mrtCount, depthBuffer: false });
      rt.textures[0].type = THREE.FloatType;
      rt.textures[0].format = THREE.RGFormat;
      rt.textures.forEach(nearest);
      return rt;
    });

    this.gMaterial = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: GBUFFER_VERT,
      fragmentShader: GBUFFER_FRAG,
    });

    this.aoMaterial = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      defines: bentNormals ? { BENT_NORMALS: 1 } : {},
      vertexShader: 'void main(){ gl_Position = vec4(position.xy, 0.0, 1.0); }',
      fragmentShader: TRACED_AO_FRAG,
      uniforms: {
        field: { value: field },
        sdfMin: { value: this.min },
        sdfInvSpan: { value: this.invSpan },
        gPos: { value: this.gTarget.textures[0] },
        gNormal: { value: this.gTarget.textures[1] },
        histTex: { value: this.histories[0].textures[0] },
        normalTex: { value: this.histories[0].textures[1] },
        bentTex: { value: bentNormals ? this.histories[0].textures[2] : null },
        prevViewProj: { value: this.prevViewProj },
        prevView: { value: this.prevView },
        curViewProj: { value: this.curViewProj },
        curView: { value: this.curView },
        invProjXY: { value: this.invProjXY },
        frame: { value: 0 },
        histMax: { value: 128 },
        settleFrames: { value: settleFrames },
        contactMode: { value: 0 },
        smoothFalloff: { value: smoothFalloff ? 1 : 0 },
        hizTex1: { value: null },
        hizTex2: { value: null },
      },
    });

    // Full-screen triangle pass (no addon dependency required)
    const quadGeo = new THREE.BufferGeometry();
    quadGeo.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
    this._fsScene = new THREE.Scene();
    this._fsCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this._fsScene.add(new THREE.Mesh(quadGeo, this.aoMaterial));
    this._quadGeo = quadGeo;

    this.contactResources = null;
    if (staticMeshes.length) {
      this.rebuildContactBVH(staticMeshes, { gridSize, bias });
    }
    this.resetHistory();
  }

  static async create(renderer, scene, camera, opts = {}) {
    const THREE = opts.THREE || (await import('three'));
    return new TracedSDFAO(renderer, scene, camera, { ...opts, THREE });
  }

  rebuildContactBVH(staticMeshes, { gridSize = 1.0, range, bias = 0.0003 } = {}) {
    const img = this.field.image;
    const autoRange = Math.max(
      0.24,
      0.12 + 0.5 * Math.hypot(this.span.x / img.width, this.span.y / img.height, this.span.z / img.depth)
    );
    const useRange = range ?? autoRange;
    const gl = this.renderer.getContext();
    const next = buildContactBVH(staticMeshes, this.T, {
      gridSize,
      range: useRange + Math.abs(bias),
      maxTextureSize: this.renderer.capabilities.maxTextureSize,
      max3DTextureSize: gl.getParameter(gl.MAX_3D_TEXTURE_SIZE),
    });
    Object.assign(this.aoMaterial.uniforms, next.uniforms, {
      contactRange: { value: useRange },
      contactBias: { value: bias },
    });
    this.contactResources?.dispose();
    this.contactResources = next;
    this.resetHistory();
    return next;
  }

  resetHistory() {
    this.frame = 0;
    this.aoMaterial.uniforms.frame.value = 0;
    for (const rt of this.histories) {
      this.renderer.setRenderTarget(rt);
      this.renderer.clear();
    }
    this.renderer.setRenderTarget(null);
  }

  /**
   * Run the half-res geometry pass + traced temporal AO pass.
   * Returns `{ aoTex, normalTex, bentTex }` for upsampling in the scene pass.
   */
  update() {
    const { renderer, scene, camera, gTarget, histories } = this;
    const cur = histories[this.histIndex];
    const prev = histories[1 - this.histIndex];
    const u = this.aoMaterial.uniforms;

    camera.updateMatrixWorld();
    u.histMax.value = camera.matrixWorld.equals(this.lastCamera) ? 128 : 32;
    this.lastCamera.copy(camera.matrixWorld);

    const prevOverride = scene.overrideMaterial;
    const prevBg = scene.background;
    const prevColor = renderer.getClearColor(new this.T.Color());
    const prevAlpha = renderer.getClearAlpha();

    scene.overrideMaterial = this.gMaterial;
    scene.background = null;
    renderer.setClearColor(0, 0);
    renderer.setRenderTarget(gTarget);
    renderer.render(scene, camera);

    scene.overrideMaterial = prevOverride;
    scene.background = prevBg;
    renderer.setClearColor(prevColor, prevAlpha);

    u.gPos.value = gTarget.textures[0];
    u.gNormal.value = gTarget.textures[1];
    u.histTex.value = prev.textures[0];
    u.normalTex.value = prev.textures[1];
    if (this.bentNormals) u.bentTex.value = prev.textures[2];
    this.curViewProj.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this.curView.copy(camera.matrixWorldInverse);
    this.invProjXY.set(1 / camera.projectionMatrix.elements[0], 1 / camera.projectionMatrix.elements[5]);

    renderer.setRenderTarget(cur);
    renderer.render(this._fsScene, this._fsCam);
    renderer.setRenderTarget(null);

    this.prevViewProj.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this.prevView.copy(camera.matrixWorldInverse);
    this.frame++;
    u.frame.value = this.frame;
    this.histIndex = 1 - this.histIndex;

    return {
      aoTex: cur.textures[0],
      normalTex: cur.textures[1],
      bentTex: this.bentNormals ? cur.textures[2] : null,
    };
  }

  dispose() {
    this.gTarget.dispose();
    for (const rt of this.histories) rt.dispose();
    this.gMaterial.dispose();
    this.aoMaterial.dispose();
    this._quadGeo.dispose();
    this.contactResources?.dispose();
  }
}
