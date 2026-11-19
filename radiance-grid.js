// 3D Direct-Lit Radiance Voxel Grid for 1-Bounce Real-Time Diffuse GI.
// Area-weighted triangle splatting (albedo + normal) with 1-voxel surface dilation
// onto a 64 x 64 x 40 RGBA8 Data3DTexture (640 KiB VRAM).
// Supports < 2 ms CPU relighting when sun direction changes using fast sphere-traced
// shadows against the scene's baked SDF on occupied surface voxels + 1-voxel dilation.

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

export function buildRadianceGrid(objects, T, sdfInfo, { dims = RADIANCE_DIMS, scale = RADIANCE_SCALE } = {}) {
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

  // Decode baked SDF once for fast CPU sun shadow queries during relight
  const sdfF32 = sdfInfo.floatData || decodeHalfVolume(sdfInfo.halfData);
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
    for (let s = 0; s < 14; s++) {
      const d = sampleSDF(ox + sdx * t, oy + sdy * t, oz + sdz * t);
      if (d > 100.0) break;
      if (d < 0.015) return 0.0;
      const pen = (8.0 * d) / t;
      if (pen < res) res = pen;
      t += d > 0.07 ? d : 0.07;
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
  const occFlag = new Uint8Array(totalVoxels);

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
        occFlag[idx] = 1;
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

  // Precompute 1-voxel dilation map: each dilated empty voxel points to its nearest surface voxel
  // (6-neighbors preferred over edge/corner neighbors) so relight copies radiance in < 0.2 ms!
  const dilDst = [];
  const dilSrc = [];
  const offs = [
    [-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1],
    [-1,-1,0],[1,-1,0],[-1,1,0],[1,1,0],[-1,0,-1],[1,0,-1],[-1,0,1],[1,0,1],[0,-1,-1],[0,1,-1],[0,-1,1],[0,1,1],
    [-1,-1,-1],[1,-1,-1],[-1,1,-1],[1,1,-1],[-1,-1,1],[1,-1,1],[-1,1,1],[1,1,1],
  ];
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      const row = z * strideZ + y * strideY;
      for (let x = 0; x < nx; x++) {
        const idx = row + x;
        if (occFlag[idx] === 1) continue;
        for (let oi = 0; oi < offs.length; oi++) {
          const xx = x + offs[oi][0], yy = y + offs[oi][1], zz = z + offs[oi][2];
          if (xx < 0 || xx >= nx || yy < 0 || yy >= ny || zz < 0 || zz >= nz) continue;
          const ni = zz * strideZ + yy * strideY + xx;
          if (occFlag[ni] === 1) {
            dilDst.push(idx);
            dilSrc.push(ni);
            break;
          }
        }
      }
    }
  }
  const dilDstArr = new Int32Array(dilDst);
  const dilSrcArr = new Int32Array(dilSrc);
  const dilCount = dilDstArr.length;

  const texData = new Uint8Array(totalVoxels * 4);
  const texture = new T.Data3DTexture(texData, nx, ny, nz);
  texture.format = T.RGBAFormat;
  texture.type = T.UnsignedByteType;
  texture.minFilter = texture.magFilter = T.LinearFilter;
  texture.generateMipmaps = false;
  texture.unpackAlignment = 1;

  const buildMs = performance.now() - t0;
  let lastRelightMs = 0;

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
    bytes: texData.byteLength,
    occupiedVoxels: surfCount,
    dilatedVoxels: dilCount,
    buildMs,
    get relightMs() { return lastRelightMs; },
    relight,
    dispose() { texture.dispose(); },
  };
}
