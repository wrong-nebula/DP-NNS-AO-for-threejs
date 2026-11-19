// 3D Direct-Lit Radiance Voxel Grid for 1-Bounce Real-Time Diffuse GI.
// Area-weighted triangle splatting (albedo + normal) with 1-voxel surface dilation
// onto a 64 x 64 x 40 RGBA8 3D volume (640 KiB VRAM).
// Supports < 0.3 ms GPU relighting (WebGL2 WebGL3DRenderTarget) when sun direction
// changes, with automatic CPU fallback when no renderer is passed (e.g. Node tests).

export const RADIANCE_DIMS = [64, 64, 40];
export const RADIANCE_SCALE = 3.5;

// Fast IEEE-754 binary16 -> float32 lookup table (built once, < 0.5 ms)
let f16Table = null;
function getF16Table() {
  if (f16Table) return f16Table;
  f16Table = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = (h >> 15) ? -1 : 1;
    const e = (h >> 10) & 31;
    const m = h & 1023;
    if (e === 0) {
      f16Table[h] = s * Math.pow(2, -14) * (m / 1024);
    } else if (e === 31) {
      f16Table[h] = m === 0 ? s * Infinity : NaN;
    } else {
      f16Table[h] = s * Math.pow(2, e - 15) * (1 + m / 1024);
    }
  }
  return f16Table;
}

function decodeHalfVolume(halfArr) {
  const lut = getF16Table();
  const out = new Float32Array(halfArr.length);
  for (let i = 0; i < halfArr.length; i++) out[i] = lut[halfArr[i]];
  return out;
}

function archiveAlbedoAt(z) {
  if (z < 0.005) return [0.37, 0.41, 0.44];
  const t = Math.max(0, Math.min(1, z / 4 + 0.14));
  return [
    0.28 + (0.64 - 0.28) * t,
    0.34 + (0.69 - 0.34) * t,
    0.43 + (0.75 - 0.43) * t,
  ];
}

export function buildRadianceGrid(objects, T, sdfInfo, {
  dims = RADIANCE_DIMS,
  scale = RADIANCE_SCALE,
  renderer = null,
  FullScreenQuad = null,
} = {}) {
  const t0 = performance.now();
  const [nx, ny, nz] = dims;
  const totalVoxels = nx * ny * nz;
  const [minX, minY, minZ] = sdfInfo.min;
  const [spanX, spanY, spanZ] = sdfInfo.span;
  const invSpanX = 1 / spanX, invSpanY = 1 / spanY, invSpanZ = 1 / spanZ;
  const voxX = spanX / nx, voxY = spanY / ny, voxZ = spanZ / nz;
  const minVox = Math.min(voxX, voxY, voxZ);
  const voxDiag = Math.hypot(voxX, voxY, voxZ);
  const sceneId = sdfInfo.sceneId || 0;
  const useGPU = Boolean(renderer && FullScreenQuad && sdfInfo.field && T.WebGL3DRenderTarget);

  // Decode baked SDF on CPU only when falling back to CPU relighting
  const sdfF32 = useGPU ? null : (sdfInfo.floatData || decodeHalfVolume(sdfInfo.halfData));
  const [snx, sny, snz] = sdfInfo.dims;
  const sdfStrideY = snx, sdfStrideZ = snx * sny;
  const sdfSx = invSpanX * snx, sdfSy = invSpanY * sny, sdfSz = invSpanZ * snz;
  const snx1 = snx - 1, sny1 = sny - 1, snz1 = snz - 1;

  function sampleSDF(x, y, z) {
    const ux = (x - minX) * sdfSx - 0.5;
    const uy = (y - minY) * sdfSy - 0.5;
    const uz = (z - minZ) * sdfSz - 0.5;
    if (ux < 0 || uy < 0 || uz < 0 || ux >= snx1 || uy >= sny1 || uz >= snz1) return 1e4;
    const ix = ux | 0, iy = uy | 0, iz = uz | 0;
    // Fast nearest-voxel lookup for relight speed (SDF is 8 cm isotropic, shadow penumbra smooths)
    return sdfF32[iz * sdfStrideZ + iy * sdfStrideY + ix];
  }

  function sdfSunShadow(px, py, pz, norX, norY, norZ, sdx, sdy, sdz) {
    const off = Math.max(0.11, voxDiag * 0.72);
    const ox = px + norX * off + sdx * 0.05;
    const oy = py + norY * off + sdy * 0.05;
    const oz = pz + norZ * off + sdz * 0.05;
    let t = 0.04;
    let res = 1.0;
    for (let s = 0; s < 16; s++) {
      const d = sampleSDF(ox + sdx * t, oy + sdy * t, oz + sdz * t);
      if (d > 100.0) break;
      if (d < 0.015) return 0.0;
      const pen = (8.0 * d) / t;
      if (pen < res) res = pen;
      t += d > 0.065 ? d : 0.065;
      if (t > 8.5) break;
    }
    return res < 0 ? 0 : (res > 1 ? 1 : res);
  }

  // Pass 1: Area-weighted triangle splatting into 64 x 64 x 40 voxel grid
  const accR = new Float32Array(totalVoxels);
  const accG = new Float32Array(totalVoxels);
  const accB = new Float32Array(totalVoxels);
  const accNX = new Float32Array(totalVoxels);
  const accNY = new Float32Array(totalVoxels);
  const accNZ = new Float32Array(totalVoxels);
  const accW = new Float32Array(totalVoxels);

  const matrix = new T.Matrix4();
  const instance = new T.Matrix4();
  const stepDist = minVox * 0.82;
  const vxScale = invSpanX * nx, vyScale = invSpanY * ny, vzScale = invSpanZ * nz;

  for (const object of objects) {
    object.updateWorldMatrix(true, false);
    const g = object.geometry;
    const pos = g.attributes.position;
    const idx = g.index;
    const colAttr = g.attributes.surfAlbedo;
    const hasPerVertexCol = colAttr && !colAttr.isInstancedBufferAttribute;
    const pArr = pos.array, pStride = pos.itemSize;
    const cArr = hasPerVertexCol ? colAttr.array : null;
    const cStride = hasPerVertexCol ? colAttr.itemSize : 0;
    const iArr = idx ? idx.array : null;
    const count = idx ? idx.count : pos.count;
    const instCount = object.isInstancedMesh ? object.count : 1;

    for (let inst = 0; inst < instCount; inst++) {
      matrix.copy(object.matrixWorld);
      if (object.isInstancedMesh) {
        object.getMatrixAt(inst, instance);
        matrix.multiply(instance);
      }
      const m = matrix.elements;
      const m0 = m[0], m1 = m[1], m2 = m[2];
      const m4 = m[4], m5 = m[5], m6 = m[6];
      const m8 = m[8], m9 = m[9], m10 = m[10];
      const m12 = m[12], m13 = m[13], m14 = m[14];

      for (let i = 0; i < count; i += 3) {
        const ia = iArr ? iArr[i] : i;
        const ib = iArr ? iArr[i + 1] : i + 1;
        const ic = iArr ? iArr[i + 2] : i + 2;
        const pa = ia * pStride, pb = ib * pStride, pc = ic * pStride;
        const ax = pArr[pa], ay = pArr[pa + 1], az = pArr[pa + 2];
        const bx = pArr[pb], by = pArr[pb + 1], bz = pArr[pb + 2];
        const cx = pArr[pc], cy = pArr[pc + 1], cz = pArr[pc + 2];

        const x0 = m0 * ax + m4 * ay + m8 * az + m12;
        const y0 = m1 * ax + m5 * ay + m9 * az + m13;
        const z0 = m2 * ax + m6 * ay + m10 * az + m14;
        const x1 = m0 * bx + m4 * by + m8 * bz + m12;
        const y1 = m1 * bx + m5 * by + m9 * bz + m13;
        const z1 = m2 * bx + m6 * by + m10 * bz + m14;
        const x2 = m0 * cx + m4 * cy + m8 * cz + m12;
        const y2 = m1 * cx + m5 * cy + m9 * cz + m13;
        const z2 = m2 * cx + m6 * cy + m10 * cz + m14;

        const e1x = x1 - x0, e1y = y1 - y0, e1z = z1 - z0;
        const e2x = x2 - x0, e2y = y2 - y0, e2z = z2 - z0;
        const fnx = e1y * e2z - e1z * e2y;
        const fny = e1z * e2x - e1x * e2z;
        const fnz = e1x * e2y - e1y * e2x;
        const area2 = Math.hypot(fnx, fny, fnz);
        if (area2 < 1e-9) continue;
        const invA2 = 1 / area2;
        const unx = fnx * invA2, uny = fny * invA2, unz = fnz * invA2;
        const area = 0.5 * area2;

        let cr, cg, cb;
        if (hasPerVertexCol && sceneId !== 0) {
          const ca = ia * cStride, cbi = ib * cStride, cc = ic * cStride;
          cr = (cArr[ca] + cArr[cbi] + cArr[cc]) / 3;
          cg = (cArr[ca + 1] + cArr[cbi + 1] + cArr[cc + 1]) / 3;
          cb = (cArr[ca + 2] + cArr[cbi + 2] + cArr[cc + 2]) / 3;
        } else {
          const midZ = (z0 + z1 + z2) / 3;
          [cr, cg, cb] = archiveAlbedoAt(midZ);
        }

        const l1 = Math.hypot(e1x, e1y, e1z);
        const l2 = Math.hypot(e2x, e2y, e2z);
        const l3 = Math.hypot(x2 - x1, y2 - y1, z2 - z1);
        const maxEdge = Math.max(l1, l2, l3);
        const steps = Math.max(1, Math.min(36, Math.ceil(maxEdge / stepDist)));

        if (steps === 1) {
          const sx = (x0 + x1 + x2) / 3;
          const sy = (y0 + y1 + y2) / 3;
          const sz = (z0 + z1 + z2) / 3;
          const vx = ((sx - minX) * vxScale) | 0;
          const vy = ((sy - minY) * vyScale) | 0;
          const vz = ((sz - minZ) * vzScale) | 0;
          if (vx >= 0 && vx < nx && vy >= 0 && vy < ny && vz >= 0 && vz < nz) {
            const o = (vz * ny + vy) * nx + vx;
            accR[o] += cr * area; accG[o] += cg * area; accB[o] += cb * area;
            accNX[o] += unx * area; accNY[o] += uny * area; accNZ[o] += unz * area;
            accW[o] += area;
          }
        } else {
          const sampleCount = (steps * (steps + 1)) * 0.5;
          const w = area / sampleCount;
          const invS = 1 / steps;
          for (let iu = 0; iu < steps; iu++) {
            for (let iv = 0; iv < steps - iu; iv++) {
              const u = (iu + 0.333333) * invS;
              const v = (iv + 0.333333) * invS;
              const sx = x0 + e1x * u + e2x * v;
              const sy = y0 + e1y * u + e2y * v;
              const sz = z0 + e1z * u + e2z * v;
              const vx = ((sx - minX) * vxScale) | 0;
              const vy = ((sy - minY) * vyScale) | 0;
              const vz = ((sz - minZ) * vzScale) | 0;
              if (vx >= 0 && vx < nx && vy >= 0 && vy < ny && vz >= 0 && vz < nz) {
                const o = (vz * ny + vy) * nx + vx;
                accR[o] += cr * w; accG[o] += cg * w; accB[o] += cb * w;
                accNX[o] += unx * w; accNY[o] += uny * w; accNZ[o] += unz * w;
                accW[o] += w;
              }
            }
          }
        }
      }
    }
  }

  // Compact list of direct surface voxels (evaluated in relight)
  const strideY = nx, strideZ = nx * ny;
  let surfCount = 0;
  for (let i = 0; i < totalVoxels; i++) if (accW[i] > 1e-8) surfCount++;

  const surfIdx = new Int32Array(surfCount);
  const surfPos = new Float32Array(surfCount * 3);
  const surfNor = new Float32Array(surfCount * 3);
  const surfAlb = new Float32Array(surfCount * 3);
  const occSurf = new Int32Array(totalVoxels);
  occSurf.fill(-1);

  let sk = 0;
  for (let z = 0; z < nz; z++) {
    const wz = minZ + (z + 0.5) * voxZ;
    for (let y = 0; y < ny; y++) {
      const wy = minY + (y + 0.5) * voxY;
      const row = z * strideZ + y * strideY;
      for (let x = 0; x < nx; x++) {
        const idx = row + x;
        const w = accW[idx];
        if (w <= 1e-8) continue;
        occSurf[idx] = sk;
        surfIdx[sk] = idx;
        const o3 = sk * 3;
        surfPos[o3] = minX + (x + 0.5) * voxX;
        surfPos[o3 + 1] = wy;
        surfPos[o3 + 2] = wz;
        const nl = Math.hypot(accNX[idx], accNY[idx], accNZ[idx]) || 1;
        surfNor[o3] = accNX[idx] / nl;
        surfNor[o3 + 1] = accNY[idx] / nl;
        surfNor[o3 + 2] = accNZ[idx] / nl;
        const iw = 1 / w;
        surfAlb[o3] = accR[idx] * iw;
        surfAlb[o3 + 1] = accG[idx] * iw;
        surfAlb[o3 + 2] = accB[idx] * iw;
        sk++;
      }
    }
  }

  // Precompute normal-aware 1-voxel dilation map into UNOCCUPIED voxels only:
  // Iterate only around occupied surface voxels (O(surfCount) instead of O(totalVoxels))
  // so build stays < 25 ms even at 72x72x48!
  const dilBestSrc = new Int32Array(totalVoxels);
  dilBestSrc.fill(-1);
  const dilBestScore = new Float32Array(totalVoxels);
  dilBestScore.fill(-1e9);
  const o6x = new Int8Array([-1,1,0,0,0,0]);
  const o6y = new Int8Array([0,0,-1,1,0,0]);
  const o6z = new Int8Array([0,0,0,0,-1,1]);
  const o12x = new Int8Array([-1,1,-1,1,-1,1,-1,1,0,0,0,0]);
  const o12y = new Int8Array([-1,-1,1,1,0,0,0,0,-1,1,-1,1]);
  const o12z = new Int8Array([0,0,0,0,-1,-1,1,1,-1,-1,1,1]);
  const o8x = new Int8Array([-1,1,-1,1,-1,1,-1,1]);
  const o8y = new Int8Array([-1,-1,1,1,-1,-1,1,1]);
  const o8z = new Int8Array([-1,-1,-1,-1,1,1,1,1]);
  const INV_SQRT2 = 0.70710678;

  for (let i = 0; i < surfCount; i++) {
    const idx = surfIdx[i];
    const z = (idx / strideZ) | 0;
    const rem = idx - z * strideZ;
    const y = (rem / strideY) | 0;
    const x = rem - y * strideY;
    const o3 = i * 3;
    const nnx = surfNor[o3], nny = surfNor[o3 + 1], nnz = surfNor[o3 + 2];
    // 6-neighbors: priority tier [20..22] for front-facing, [10..12] for back/interior (prefer non-upward)
    for (let k = 0; k < 6; k++) {
      const dx = o6x[k], dy = o6y[k], dz = o6z[k];
      const xx = x + dx, yy = y + dy, zz = z + dz;
      if (xx < 0 || xx >= nx || yy < 0 || yy >= ny || zz < 0 || zz >= nz) continue;
      const ni = zz * strideZ + yy * strideY + xx;
      if (occSurf[ni] >= 0) continue;
      const d = nnx * dx + nny * dy + nnz * dz;
      const sc = d > 0.15 ? 20.0 + d : 10.0 - nnz;
      if (sc > dilBestScore[ni]) { dilBestScore[ni] = sc; dilBestSrc[ni] = idx; }
    }
    // 12-edge neighbors: priority tier [5..7] front-facing, [2..4] fallback
    for (let k = 0; k < 12; k++) {
      const dx = o12x[k], dy = o12y[k], dz = o12z[k];
      const xx = x + dx, yy = y + dy, zz = z + dz;
      if (xx < 0 || xx >= nx || yy < 0 || yy >= ny || zz < 0 || zz >= nz) continue;
      const ni = zz * strideZ + yy * strideY + xx;
      if (occSurf[ni] >= 0) continue;
      const d = (nnx * dx + nny * dy + nnz * dz) * INV_SQRT2;
      const sc = d > 0.2 ? 5.0 + d : 2.0 - nnz;
      if (sc > dilBestScore[ni]) { dilBestScore[ni] = sc; dilBestSrc[ni] = idx; }
    }
    // 8-corner neighbors: priority tier [0..1]
    for (let k = 0; k < 8; k++) {
      const xx = x + o8x[k], yy = y + o8y[k], zz = z + o8z[k];
      if (xx < 0 || xx >= nx || yy < 0 || yy >= ny || zz < 0 || zz >= nz) continue;
      const ni = zz * strideZ + yy * strideY + xx;
      if (occSurf[ni] >= 0) continue;
      const sc = 0.5 - nnz * 0.25;
      if (sc > dilBestScore[ni]) { dilBestScore[ni] = sc; dilBestSrc[ni] = idx; }
    }
  }
  let dilCount = 0;
  for (let i = 0; i < totalVoxels; i++) if (dilBestSrc[i] >= 0) dilCount++;
  const dilDstArr = new Int32Array(dilCount);
  const dilSrcArr = new Int32Array(dilCount);
  let dk = 0;
  for (let i = 0; i < totalVoxels; i++) {
    const s = dilBestSrc[i];
    if (s >= 0) { dilDstArr[dk] = i; dilSrcArr[dk] = s; dk++; }
  }

  let lastRelightMs = 0;

  if (useGPU) {
    // Pack static dilated albedo (.a = occupied/dilated mask) & normal (.a = source-voxel offset code 0..26)
    // so GPU relight evaluates both occupied surface voxels and exact 1-voxel dilated copies in ONE 40-slice pass.
    const albData = new Uint8Array(totalVoxels * 4);
    const norData = new Uint8Array(totalVoxels * 4);
    for (let i = 0; i < surfCount; i++) {
      const o3 = i * 3;
      const o4 = surfIdx[i] * 4;
      const ar = surfAlb[o3] * 255, ag = surfAlb[o3 + 1] * 255, ab = surfAlb[o3 + 2] * 255;
      albData[o4] = ar >= 255 ? 255 : (ar <= 0 ? 0 : (ar + 0.5) | 0);
      albData[o4 + 1] = ag >= 255 ? 255 : (ag <= 0 ? 0 : (ag + 0.5) | 0);
      albData[o4 + 2] = ab >= 255 ? 255 : (ab <= 0 ? 0 : (ab + 0.5) | 0);
      albData[o4 + 3] = 255;
      const nxu = (surfNor[o3] * 0.5 + 0.5) * 255;
      const nyu = (surfNor[o3 + 1] * 0.5 + 0.5) * 255;
      const nzu = (surfNor[o3 + 2] * 0.5 + 0.5) * 255;
      norData[o4] = nxu >= 255 ? 255 : (nxu <= 0 ? 0 : (nxu + 0.5) | 0);
      norData[o4 + 1] = nyu >= 255 ? 255 : (nyu <= 0 ? 0 : (nyu + 0.5) | 0);
      norData[o4 + 2] = nzu >= 255 ? 255 : (nzu <= 0 ? 0 : (nzu + 0.5) | 0);
      norData[o4 + 3] = 13; // dx=0, dy=0, dz=0 => (0+1) + (0+1)*3 + (0+1)*9
    }
    for (let i = 0; i < dilCount; i++) {
      const dst = dilDstArr[i];
      const src = dilSrcArr[i];
      const dzD = (dst / strideZ) | 0;
      const remD = dst - dzD * strideZ;
      const dyD = (remD / strideY) | 0;
      const dxD = remD - dyD * strideY;
      const dzS = (src / strideZ) | 0;
      const remS = src - dzS * strideZ;
      const dyS = (remS / strideY) | 0;
      const dxS = remS - dyS * strideY;
      const code = (dxS - dxD + 1) + (dyS - dyD + 1) * 3 + (dzS - dzD + 1) * 9;
      const d4 = dst * 4, s4 = src * 4;
      albData[d4] = albData[s4];
      albData[d4 + 1] = albData[s4 + 1];
      albData[d4 + 2] = albData[s4 + 2];
      albData[d4 + 3] = 255;
      norData[d4] = norData[s4];
      norData[d4 + 1] = norData[s4 + 1];
      norData[d4 + 2] = norData[s4 + 2];
      norData[d4 + 3] = code;
    }

    const albedoTex = new T.Data3DTexture(albData, nx, ny, nz);
    albedoTex.format = T.RGBAFormat;
    albedoTex.type = T.UnsignedByteType;
    albedoTex.minFilter = albedoTex.magFilter = T.NearestFilter;
    albedoTex.generateMipmaps = false;
    albedoTex.unpackAlignment = 1;
    albedoTex.needsUpdate = true;

    const normalTex = new T.Data3DTexture(norData, nx, ny, nz);
    normalTex.format = T.RGBAFormat;
    normalTex.type = T.UnsignedByteType;
    normalTex.minFilter = normalTex.magFilter = T.NearestFilter;
    normalTex.generateMipmaps = false;
    normalTex.unpackAlignment = 1;
    normalTex.needsUpdate = true;

    const targetA = new T.WebGL3DRenderTarget(nx, ny, nz, { depthBuffer: false, stencilBuffer: false });
    targetA.texture.format = T.RGBAFormat;
    targetA.texture.type = T.UnsignedByteType;
    targetA.texture.minFilter = T.LinearMipmapLinearFilter;
    targetA.texture.magFilter = T.LinearFilter;
    targetA.texture.generateMipmaps = false;

    let targetB = null;
    function ensureTargetB() {
      if (targetB) return targetB;
      targetB = new T.WebGL3DRenderTarget(nx, ny, nz, { depthBuffer: false, stencilBuffer: false });
      targetB.texture.format = T.RGBAFormat;
      targetB.texture.type = T.UnsignedByteType;
      targetB.texture.minFilter = T.LinearMipmapLinearFilter;
      targetB.texture.magFilter = T.LinearFilter;
      targetB.texture.generateMipmaps = false;
      return targetB;
    }

    function generate3DMipmaps(rt) {
      const gl = renderer.getContext();
      const props = renderer.properties.get(rt.texture);
      const glTex = props && props.__webglTexture;
      if (!glTex) return;
      const prevTex = gl.getParameter(gl.TEXTURE_BINDING_3D);
      gl.bindTexture(gl.TEXTURE_3D, glTex);
      gl.generateMipmap(gl.TEXTURE_3D);
      gl.bindTexture(gl.TEXTURE_3D, prevTex);
    }

    const relightMat = new T.ShaderMaterial({
      glslVersion: T.GLSL3,
      depthTest: false,
      depthWrite: false,
      uniforms: {
        albedoTex: { value: albedoTex },
        normalTex: { value: normalTex },
        sdfField: { value: sdfInfo.field },
        sdfMin: { value: new T.Vector3(minX, minY, minZ) },
        sdfInvSpan: { value: new T.Vector3(invSpanX, invSpanY, invSpanZ) },
        voxSize: { value: new T.Vector3(voxX, voxY, voxZ) },
        sunDir: { value: new T.Vector3(0, 0, 1) },
        sunColor: { value: new T.Vector3(3.10, 2.90, 2.55) },
        skyColor: { value: new T.Vector3(0.56, 0.64, 0.76) },
        emissivePos: { value: new T.Vector3(0, 0, -100) },
        emissiveColor: { value: new T.Vector3(0, 0, 0) },
        emissiveRadius: { value: 0.18 },
        invScale: { value: 1 / scale },
        voxOff: { value: Math.max(0.11, voxDiag * 0.72) },
        voxDiag: { value: voxDiag },
        layer: { value: 0 },
      },
      vertexShader: 'void main(){gl_Position=vec4(position.xy,0.,1.);}',
      fragmentShader: `precision highp float;
      precision highp sampler3D;
      uniform sampler3D albedoTex;
      uniform sampler3D normalTex;
      uniform sampler3D sdfField;
      uniform vec3 sdfMin;
      uniform vec3 sdfInvSpan;
      uniform vec3 voxSize;
      uniform vec3 sunDir;
      uniform vec3 sunColor;
      uniform vec3 skyColor;
      uniform vec3 emissivePos;
      uniform vec3 emissiveColor;
      uniform float emissiveRadius;
      uniform float invScale;
      uniform float voxOff;
      uniform float voxDiag;
      uniform int layer;
      out vec4 o;
      void main(){
        ivec3 v = ivec3(gl_FragCoord.xy, layer);
        vec3 voxPos = sdfMin + vec3(gl_FragCoord.xy, float(layer) + 0.5) * voxSize;
        bool hasEmissive = dot(emissiveColor, emissiveColor) > 1e-5;
        vec4 alb = texelFetch(albedoTex, v, 0);
        if(alb.a < 0.5){
          // Inject floating emissive orb directly into open-air radiance voxels so GI rays pick up its glow
          if(hasEmissive){
            float dOrb = length(voxPos - emissivePos);
            float rInfl = emissiveRadius + voxDiag * 1.05;
            if(dOrb < rInfl){
              float glow = smoothstep(rInfl, emissiveRadius * 0.45, dOrb);
              o = vec4(clamp(emissiveColor * (glow * 1.35) * invScale, 0.0, 1.0), 1.0);
              return;
            }
          }
          o = vec4(0.);
          return;
        }
        vec4 nSample = texelFetch(normalTex, v, 0);
        vec3 nor = normalize(nSample.rgb * 2.0 - 1.0);
        int code = int(floor(nSample.a * 255.0 + 0.5));
        int dxi = (code % 3) - 1;
        int dyi = ((code / 3) % 3) - 1;
        int dzi = (code / 9) - 1;
        vec3 srcPos = voxPos + vec3(float(dxi), float(dyi), float(dzi)) * voxSize;
        float ndotl = dot(nor, sunDir);
        float sh = 0.0;
        if(ndotl > 0.001){
          vec3 ro = srcPos + nor * voxOff + sunDir * 0.05;
          float t = 0.04;
          float res = 1.0;
          for(int s = 0; s < 16; s++){
            vec3 uv = (ro + sunDir * t - sdfMin) * sdfInvSpan;
            if(any(lessThan(uv, vec3(0.))) || any(greaterThan(uv, vec3(1.)))) break;
            float d = texture(sdfField, uv).r;
            if(d < 0.015){ res = 0.0; break; }
            res = min(res, (8.0 * d) / t);
            if(res < 0.005){ res = 0.0; break; }
            t += max(d, 0.065);
            if(t > 8.5) break;
          }
          sh = clamp(res, 0.0, 1.0);
        }
        float dSun = max(0.0, ndotl) * sh;
        float skyF = 0.42 + 0.32 * max(0.0, nor.z);
        vec3 irr = sunColor * dSun + skyColor * skyF;
        if(hasEmissive){
          vec3 toOrb = emissivePos - srcPos;
          float orbDist = length(toOrb);
          vec3 lDir = toOrb / max(orbDist, 1e-4);
          float ndlOrb = max(0.0, dot(nor, lDir));
          if(ndlOrb > 0.001){
            float maxT = max(0.0, orbDist - emissiveRadius - 0.05);
            float orbSh = 1.0;
            if(maxT > 0.06){
              vec3 ro = srcPos + nor * voxOff;
              float t = 0.05;
              for(int s = 0; s < 12; s++){
                vec3 uv = (ro + lDir * t - sdfMin) * sdfInvSpan;
                if(any(lessThan(uv, vec3(0.))) || any(greaterThan(uv, vec3(1.)))) break;
                float d = texture(sdfField, uv).r;
                if(d < 0.015){ orbSh = 0.0; break; }
                orbSh = min(orbSh, (6.0 * d) / t);
                if(orbSh < 0.005){ orbSh = 0.0; break; }
                t += max(d, 0.08);
                if(t >= maxT) break;
              }
            }
            float atten = 1.0 / (orbDist * orbDist + 0.25);
            irr += emissiveColor * (ndlOrb * atten * clamp(orbSh, 0.0, 1.0));
          }
        }
        o = vec4(clamp(alb.rgb * irr * invScale, 0.0, 1.0), 1.0);
      }`,
    });
    const relightQuad = new FullScreenQuad(relightMat);

    // Pass 2: Lumen-style GPU multi-bounce propagation (targetA -> targetB).
    // Gathers 6 hemisphere-aligned short SDF rays per occupied/dilated voxel and adds
    // physically bounded indirect bounce (bounceWeight < 0.85) to direct radiance.
    // Runs ONLY during relight() -> 0.00 ms per-frame cost when lighting is static!
    const bounceMat = new T.ShaderMaterial({
      glslVersion: T.GLSL3,
      depthTest: false,
      depthWrite: false,
      uniforms: {
        albedoTex: { value: albedoTex },
        normalTex: { value: normalTex },
        srcVol: { value: targetA.texture },
        sdfField: { value: sdfInfo.field },
        sdfMin: { value: new T.Vector3(minX, minY, minZ) },
        sdfInvSpan: { value: new T.Vector3(invSpanX, invSpanY, invSpanZ) },
        voxSize: { value: new T.Vector3(voxX, voxY, voxZ) },
        scale: { value: scale },
        invScale: { value: 1 / scale },
        voxOff: { value: Math.max(0.11, voxDiag * 0.72) },
        bounceWeight: { value: 0.72 },
        layer: { value: 0 },
      },
      vertexShader: 'void main(){gl_Position=vec4(position.xy,0.,1.);}',
      fragmentShader: `precision highp float;
      precision highp sampler3D;
      uniform sampler3D albedoTex;
      uniform sampler3D normalTex;
      uniform sampler3D srcVol;
      uniform sampler3D sdfField;
      uniform vec3 sdfMin;
      uniform vec3 sdfInvSpan;
      uniform vec3 voxSize;
      uniform float scale;
      uniform float invScale;
      uniform float voxOff;
      uniform float bounceWeight;
      uniform int layer;
      out vec4 o;
      // 6 cosine-weighted hemisphere directions (1 zenith + 5 equiangular ring)
      const vec3 HDIRS[6] = vec3[6](
        vec3( 0.0000,  0.0000, 1.0000),
        vec3( 0.7800,  0.0000, 0.6258),
        vec3( 0.2410,  0.7418, 0.6258),
        vec3(-0.6310,  0.4585, 0.6258),
        vec3(-0.6310, -0.4585, 0.6258),
        vec3( 0.2410, -0.7418, 0.6258)
      );
      void main(){
        ivec3 v = ivec3(gl_FragCoord.xy, layer);
        vec4 directTap = texelFetch(srcVol, v, 0);
        vec4 alb = texelFetch(albedoTex, v, 0);
        if(alb.a < 0.5){
          // Preserve emissive open-air voxels verbatim
          o = directTap;
          return;
        }
        vec4 nSample = texelFetch(normalTex, v, 0);
        vec3 nor = normalize(nSample.rgb * 2.0 - 1.0);
        int code = int(floor(nSample.a * 255.0 + 0.5));
        int dxi = (code % 3) - 1;
        int dyi = ((code / 3) % 3) - 1;
        int dzi = (code / 9) - 1;
        vec3 voxPos = sdfMin + vec3(gl_FragCoord.xy, float(layer) + 0.5) * voxSize;
        vec3 srcPos = voxPos + vec3(float(dxi), float(dyi), float(dzi)) * voxSize;
        vec3 a = abs(nor.x) < 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
        vec3 tanX = normalize(cross(nor, a));
        vec3 tanY = cross(nor, tanX);
        vec3 ro = srcPos + nor * voxOff;
        vec3 indirect = vec3(0.0);
        for(int i = 0; i < 6; i++){
          vec3 hd = HDIRS[i];
          vec3 w = tanX * hd.x + tanY * hd.y + nor * hd.z;
          float t = voxOff * 1.15;
          for(int s = 0; s < 5; s++){
            vec3 p = ro + w * t;
            vec3 uv = (p - sdfMin) * sdfInvSpan;
            if(any(lessThan(uv, vec3(0.))) || any(greaterThan(uv, vec3(1.)))) break;
            float d = texture(sdfField, uv).r;
            if(d < 0.065){
              vec3 hitUV = (p + w * max(d, 0.0) - sdfMin) * sdfInvSpan;
              indirect += textureLod(srcVol, clamp(hitUV, vec3(0.0), vec3(1.0)), 0.0).rgb * scale;
              break;
            }
            t += max(d, 0.24);
            if(t > 6.0) break;
          }
        }
        indirect *= (1.0 / 6.0);
        vec3 directRad = directTap.rgb * scale;
        vec3 totalRad = directRad + bounceWeight * alb.rgb * indirect;
        o = vec4(clamp(totalRad * invScale, 0.0, 1.0), 1.0);
      }`,
    });
    const bounceQuad = new FullScreenQuad(bounceMat);

    const buildMs = performance.now() - t0;
    let activeTarget = targetA;
    let lastMultiBounce = false;

    function relight(sunDir, {
      sunColor = [3.10, 2.90, 2.55],
      skyColor = [0.56, 0.64, 0.76],
      multiBounce = false,
      bouncePasses = 1,
      bounceWeight = 0.72,
      emissive = null,
    } = {}) {
      const rt0 = performance.now();
      const u = relightMat.uniforms;
      u.sunDir.value.set(sunDir[0], sunDir[1], sunDir[2]);
      u.sunColor.value.set(sunColor[0], sunColor[1], sunColor[2]);
      u.skyColor.value.set(skyColor[0], skyColor[1], skyColor[2]);
      if (emissive && emissive.color && (emissive.color[0] > 0 || emissive.color[1] > 0 || emissive.color[2] > 0)) {
        u.emissivePos.value.set(emissive.pos[0], emissive.pos[1], emissive.pos[2]);
        u.emissiveColor.value.set(emissive.color[0], emissive.color[1], emissive.color[2]);
        u.emissiveRadius.value = emissive.radius || 0.18;
      } else {
        u.emissivePos.value.set(0, 0, -100);
        u.emissiveColor.value.set(0, 0, 0);
      }
      const prevRT = renderer.getRenderTarget();
      for (let z = 0; z < nz; z++) {
        u.layer.value = z;
        renderer.setRenderTarget(targetA, z);
        relightQuad.render(renderer);
      }
      generate3DMipmaps(targetA);
      if (multiBounce) {
        const dst = ensureTargetB();
        const bu = bounceMat.uniforms;
        bu.bounceWeight.value = Math.max(0, Math.min(0.84, bounceWeight));
        const nPasses = Math.max(1, Math.min(2, bouncePasses | 0));
        let srcRT = targetA, dstRT = dst;
        for (let bp = 0; bp < nPasses; bp++) {
          bu.srcVol.value = srcRT.texture;
          for (let z = 0; z < nz; z++) {
            bu.layer.value = z;
            renderer.setRenderTarget(dstRT, z);
            bounceQuad.render(renderer);
          }
          generate3DMipmaps(dstRT);
          const tmp = srcRT; srcRT = dstRT; dstRT = tmp;
        }
        activeTarget = srcRT;
        lastMultiBounce = true;
      } else {
        activeTarget = targetA;
        lastMultiBounce = false;
      }
      renderer.setRenderTarget(prevRT);
      lastRelightMs = performance.now() - rt0;
      return lastRelightMs;
    }

    return {
      get texture() { return activeTarget.texture; },
      dims,
      scale,
      voxStep: Math.max(voxX, voxY, voxZ),
      get bytes() { return totalVoxels * 4 * (targetB ? 2 : 1); },
      occupiedVoxels: surfCount,
      dilatedVoxels: dilCount,
      buildMs,
      gpu: true,
      get multiBounce() { return lastMultiBounce; },
      get relightMs() { return lastRelightMs; },
      relight,
      dispose() {
        targetA.dispose();
        targetB?.dispose();
        albedoTex.dispose();
        normalTex.dispose();
        relightMat.dispose();
        relightQuad.dispose();
        bounceMat.dispose();
        bounceQuad.dispose();
      },
    };
  }

  // CPU fallback (for Node unit tests or when no WebGL2 renderer is passed)
  const texData = new Uint8Array(totalVoxels * 4);
  const texture = new T.Data3DTexture(texData, nx, ny, nz);
  texture.format = T.RGBAFormat;
  texture.type = T.UnsignedByteType;
  texture.minFilter = texture.magFilter = T.LinearFilter;
  texture.generateMipmaps = false;
  texture.unpackAlignment = 1;

  const buildMs = performance.now() - t0;

  function relight(sunDir, {
    sunColor = [3.10, 2.90, 2.55],
    skyColor = [0.56, 0.64, 0.76],
  } = {}) {
    const rt0 = performance.now();
    const sdx = sunDir[0], sdy = sunDir[1], sdz = sunDir[2];
    const sr = sunColor[0], sg = sunColor[1], sb = sunColor[2];
    const kr = skyColor[0], kg = skyColor[1], kb = skyColor[2];
    const invScale = 255 / scale;

    for (let i = 0; i < surfCount; i++) {
      const o3 = i * 3;
      const nnx = surfNor[o3], nny = surfNor[o3 + 1], nnz = surfNor[o3 + 2];
      const ndotl = nnx * sdx + nny * sdy + nnz * sdz;
      let sh = 0;
      if (ndotl > 0.001) {
        sh = sdfSunShadow(
          surfPos[o3], surfPos[o3 + 1], surfPos[o3 + 2],
          nnx, nny, nnz,
          sdx, sdy, sdz
        );
      }
      const dSun = ndotl > 0 ? ndotl * sh : 0;
      const skyF = 0.42 + 0.32 * (nnz > 0 ? nnz : 0);
      const irR = sr * dSun + kr * skyF;
      const irG = sg * dSun + kg * skyF;
      const irB = sb * dSun + kb * skyF;

      const lr = surfAlb[o3] * irR * invScale;
      const lg = surfAlb[o3 + 1] * irG * invScale;
      const lb = surfAlb[o3 + 2] * irB * invScale;

      const o4 = surfIdx[i] * 4;
      texData[o4] = lr >= 255 ? 255 : (lr <= 0 ? 0 : lr | 0);
      texData[o4 + 1] = lg >= 255 ? 255 : (lg <= 0 ? 0 : lg | 0);
      texData[o4 + 2] = lb >= 255 ? 255 : (lb <= 0 ? 0 : lb | 0);
      texData[o4 + 3] = 255;
    }
    // Fast 1-voxel dilation copy so trilinear lookups at hit points never bleed black
    for (let i = 0; i < dilCount; i++) {
      const d4 = dilDstArr[i] * 4;
      const s4 = dilSrcArr[i] * 4;
      texData[d4] = texData[s4];
      texData[d4 + 1] = texData[s4 + 1];
      texData[d4 + 2] = texData[s4 + 2];
      texData[d4 + 3] = 255;
    }
    texture.needsUpdate = true;
    lastRelightMs = performance.now() - rt0;
    return lastRelightMs;
  }

  return {
    texture,
    dims,
    scale,
    voxStep: Math.max(voxX, voxY, voxZ),
    bytes: texData.byteLength,
    occupiedVoxels: surfCount,
    dilatedVoxels: dilCount,
    buildMs,
    gpu: false,
    get relightMs() { return lastRelightMs; },
    relight,
    dispose() { texture.dispose(); },
  };
}
