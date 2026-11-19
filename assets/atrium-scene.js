// Procedural Curved Atrium / Rotunda scene (zero external dependencies).
// Generates watertight, non-axis-aligned curved triangle meshes (fluted columns,
// torus moldings, vaulted arches, curved apse wall, tiered rotunda pedestal,
// torus knot sculpture, lathed teapot & amphorae, curved balustrades, angled benches).
// Shared identically between tools/gen_atrium.mjs (SDF bake & ray-traced reference)
// and index.html (Three.js WebGL2 runtime).

export const ATRIUM_SDF_MIN = [-6.5, -6.5, -0.5];
export const ATRIUM_SDF_SPAN = [13.0, 13.0, 6.5];
export const ATRIUM_SDF_DIMS = [112, 112, 56];

export const ATRIUM_CAMERA = {
  position: [0.35, -9.2, 3.65],
  target: [0.0, 0.35, 1.55],
};

class MeshBuilder {
  constructor() {
    this.pos = [];
    this.nor = [];
    this.col = [];
    this.idx = [];
  }

  addVertex(x, y, z, nx, ny, nz, r, g, b) {
    const l = Math.hypot(nx, ny, nz) || 1;
    const id = this.pos.length / 3;
    this.pos.push(x, y, z);
    this.nor.push(nx / l, ny / l, nz / l);
    this.col.push(r, g, b);
    return id;
  }

  addTri(a, b, c) {
    this.idx.push(a, b, c);
  }

  addQuad(a, b, c, d) {
    this.idx.push(a, b, c, a, c, d);
  }

  // Closed cylinder / frustum along Z from z0 to z1, optional fluting
  addCylinder(cx, cy, z0, z1, r0, r1, segs, col, { flutes = 0, fluteDepth = 0, angleOffset = 0 } = {}) {
    const [cr, cg, cb] = col;
    const ring0 = [];
    const ring1 = [];
    const radAt = (t, baseR) => {
      if (!flutes) return baseR;
      const s = Math.max(0, Math.sin(flutes * t));
      return baseR - fluteDepth * s * s;
    };
    const eps = 1e-3;
    const dz = z1 - z0;
    for (let i = 0; i <= segs; i++) {
      const a = angleOffset + (i / segs) * Math.PI * 2;
      const ca = Math.cos(a), sa = Math.sin(a);
      const rad0 = radAt(a, r0), rad1 = radAt(a, r1);
      // Tangent along angle for accurate fluted normal
      const aP = a + eps, aM = a - eps;
      const rP = radAt(aP, 0.5 * (r0 + r1)), rM = radAt(aM, 0.5 * (r0 + r1));
      const tx = (rP * Math.cos(aP) - rM * Math.cos(aM)) / (2 * eps);
      const ty = (rP * Math.sin(aP) - rM * Math.sin(aM)) / (2 * eps);
      const dr = rad1 - rad0;
      // Side normal = cross(tangent_theta, tangent_z)
      const tzX = dr * ca, tzY = dr * sa, tzZ = dz;
      const nx = ty * tzZ - 0 * tzY;
      const ny = 0 * tzX - tx * tzZ;
      const nz = tx * tzY - ty * tzX;
      ring0.push(this.addVertex(cx + rad0 * ca, cy + rad0 * sa, z0, nx, ny, nz, cr, cg, cb));
      ring1.push(this.addVertex(cx + rad1 * ca, cy + rad1 * sa, z1, nx, ny, nz, cr, cg, cb));
    }
    for (let i = 0; i < segs; i++) {
      this.addQuad(ring0[i], ring0[i + 1], ring1[i + 1], ring1[i]);
    }
    // Bottom cap (normal -Z)
    const botCenter = this.addVertex(cx, cy, z0, 0, 0, -1, cr, cg, cb);
    const botRing = [];
    for (let i = 0; i < segs; i++) {
      const a = angleOffset + (i / segs) * Math.PI * 2;
      const rad0 = radAt(a, r0);
      botRing.push(this.addVertex(cx + rad0 * Math.cos(a), cy + rad0 * Math.sin(a), z0, 0, 0, -1, cr, cg, cb));
    }
    for (let i = 0; i < segs; i++) {
      this.addTri(botCenter, botRing[(i + 1) % segs], botRing[i]);
    }
    // Top cap (normal +Z)
    const topCenter = this.addVertex(cx, cy, z1, 0, 0, 1, cr, cg, cb);
    const topRing = [];
    for (let i = 0; i < segs; i++) {
      const a = angleOffset + (i / segs) * Math.PI * 2;
      const rad1 = radAt(a, r1);
      topRing.push(this.addVertex(cx + rad1 * Math.cos(a), cy + rad1 * Math.sin(a), z1, 0, 0, 1, cr, cg, cb));
    }
    for (let i = 0; i < segs; i++) {
      this.addTri(topCenter, topRing[i], topRing[(i + 1) % segs]);
    }
  }

  // Closed torus in XY plane centered at (cx, cy, cz)
  addTorus(cx, cy, cz, R, r, uSegs, vSegs, col) {
    const [cr, cg, cb] = col;
    const grid = [];
    for (let i = 0; i <= uSegs; i++) {
      const u = (i / uSegs) * Math.PI * 2;
      const cu = Math.cos(u), su = Math.sin(u);
      const row = [];
      for (let j = 0; j <= vSegs; j++) {
        const v = (j / vSegs) * Math.PI * 2;
        const cv = Math.cos(v), sv = Math.sin(v);
        const rr = R + r * cv;
        row.push(this.addVertex(
          cx + rr * cu, cy + rr * su, cz + r * sv,
          cv * cu, cv * su, sv, cr, cg, cb
        ));
      }
      grid.push(row);
    }
    for (let i = 0; i < uSegs; i++) {
      for (let j = 0; j < vSegs; j++) {
        this.addQuad(grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]);
      }
    }
  }

  // Closed annular wall or arc segment: radius [rIn, rOut], z [z0, z1], angle [a0, a1]
  addArcWall(rIn, rOut, z0, z1, a0, a1, segs, col) {
    const [cr, cg, cb] = col;
    const full = Math.abs((a1 - a0) - Math.PI * 2) < 1e-4;
    // Outer wall (+radial)
    const out0 = [], out1 = [];
    // Inner wall (-radial)
    const in0 = [], in1 = [];
    // Top face (+Z)
    const topIn = [], topOut = [];
    // Bottom face (-Z)
    const botIn = [], botOut = [];
    for (let i = 0; i <= segs; i++) {
      const a = a0 + (i / segs) * (a1 - a0);
      const ca = Math.cos(a), sa = Math.sin(a);
      out0.push(this.addVertex(rOut * ca, rOut * sa, z0, ca, sa, 0, cr, cg, cb));
      out1.push(this.addVertex(rOut * ca, rOut * sa, z1, ca, sa, 0, cr, cg, cb));
      in0.push(this.addVertex(rIn * ca, rIn * sa, z0, -ca, -sa, 0, cr, cg, cb));
      in1.push(this.addVertex(rIn * ca, rIn * sa, z1, -ca, -sa, 0, cr, cg, cb));
      topIn.push(this.addVertex(rIn * ca, rIn * sa, z1, 0, 0, 1, cr, cg, cb));
      topOut.push(this.addVertex(rOut * ca, rOut * sa, z1, 0, 0, 1, cr, cg, cb));
      botIn.push(this.addVertex(rIn * ca, rIn * sa, z0, 0, 0, -1, cr, cg, cb));
      botOut.push(this.addVertex(rOut * ca, rOut * sa, z0, 0, 0, -1, cr, cg, cb));
    }
    for (let i = 0; i < segs; i++) {
      this.addQuad(out0[i], out0[i + 1], out1[i + 1], out1[i]);
      this.addQuad(in0[i + 1], in0[i], in1[i], in1[i + 1]);
      this.addQuad(topOut[i], topOut[i + 1], topIn[i + 1], topIn[i]);
      this.addQuad(botIn[i], botIn[i + 1], botOut[i + 1], botOut[i]);
    }
    if (!full) {
      // End cap at a0 (normal = +sin(a0), -cos(a0), 0)
      const c0 = Math.cos(a0), s0 = Math.sin(a0);
      const e0a = this.addVertex(rIn * c0, rIn * s0, z0, s0, -c0, 0, cr, cg, cb);
      const e0b = this.addVertex(rOut * c0, rOut * s0, z0, s0, -c0, 0, cr, cg, cb);
      const e0c = this.addVertex(rOut * c0, rOut * s0, z1, s0, -c0, 0, cr, cg, cb);
      const e0d = this.addVertex(rIn * c0, rIn * s0, z1, s0, -c0, 0, cr, cg, cb);
      this.addQuad(e0a, e0b, e0c, e0d);
      // End cap at a1 (normal = -sin(a1), +cos(a1), 0)
      const c1 = Math.cos(a1), s1 = Math.sin(a1);
      const e1a = this.addVertex(rOut * c1, rOut * s1, z0, -s1, c1, 0, cr, cg, cb);
      const e1b = this.addVertex(rIn * c1, rIn * s1, z0, -s1, c1, 0, cr, cg, cb);
      const e1c = this.addVertex(rIn * c1, rIn * s1, z1, -s1, c1, 0, cr, cg, cb);
      const e1d = this.addVertex(rOut * c1, rOut * s1, z1, -s1, c1, 0, cr, cg, cb);
      this.addQuad(e1a, e1b, e1c, e1d);
    }
  }

  // Closed UV sphere
  addSphere(cx, cy, cz, r, uSegs, vSegs, col) {
    const [cr, cg, cb] = col;
    const grid = [];
    for (let j = 0; j <= vSegs; j++) {
      const phi = (j / vSegs) * Math.PI;
      const sp = Math.sin(phi), cp = Math.cos(phi);
      const row = [];
      for (let i = 0; i <= uSegs; i++) {
        const theta = (i / uSegs) * Math.PI * 2;
        const nx = sp * Math.cos(theta), ny = sp * Math.sin(theta), nz = cp;
        row.push(this.addVertex(cx + r * nx, cy + r * ny, cz + r * nz, nx, ny, nz, cr, cg, cb));
      }
      grid.push(row);
    }
    for (let j = 0; j < vSegs; j++) {
      for (let i = 0; i < uSegs; i++) {
        if (j > 0) this.addTri(grid[j][i], grid[j + 1][i], grid[j][i + 1]);
        if (j < vSegs - 1) this.addTri(grid[j][i + 1], grid[j + 1][i], grid[j + 1][i + 1]);
      }
    }
  }

  // Closed surface of revolution around Z axis from profile [[r, z], ...] (bottom to top)
  addLathe(cx, cy, cz, profile, segs, col) {
    const [cr, cg, cb] = col;
    const nP = profile.length;
    const rings = [];
    for (let p = 0; p < nP; p++) {
      const [r, z] = profile[p];
      const pPrev = profile[Math.max(0, p - 1)];
      const pNext = profile[Math.min(nP - 1, p + 1)];
      const dr = pNext[0] - pPrev[0];
      const dz = pNext[1] - pPrev[1];
      // Outward normal in (r, z) plane is (dz, -dr)
      const nr = dz, nz = -dr;
      const row = [];
      for (let i = 0; i <= segs; i++) {
        const a = (i / segs) * Math.PI * 2;
        const ca = Math.cos(a), sa = Math.sin(a);
        row.push(this.addVertex(cx + r * ca, cy + r * sa, cz + z, nr * ca, nr * sa, nz, cr, cg, cb));
      }
      rings.push(row);
    }
    for (let p = 0; p < nP - 1; p++) {
      for (let i = 0; i < segs; i++) {
        this.addQuad(rings[p][i], rings[p][i + 1], rings[p + 1][i + 1], rings[p + 1][i]);
      }
    }
    // Close bottom if r > 0
    if (profile[0][0] > 1e-4) {
      const [r0, z0] = profile[0];
      const cBot = this.addVertex(cx, cy, cz + z0, 0, 0, -1, cr, cg, cb);
      const bRing = [];
      for (let i = 0; i < segs; i++) {
        const a = (i / segs) * Math.PI * 2;
        bRing.push(this.addVertex(cx + r0 * Math.cos(a), cy + r0 * Math.sin(a), cz + z0, 0, 0, -1, cr, cg, cb));
      }
      for (let i = 0; i < segs; i++) this.addTri(cBot, bRing[(i + 1) % segs], bRing[i]);
    }
    // Close top if r > 0
    const last = profile[nP - 1];
    if (last[0] > 1e-4) {
      const [r1, z1] = last;
      const cTop = this.addVertex(cx, cy, cz + z1, 0, 0, 1, cr, cg, cb);
      const tRing = [];
      for (let i = 0; i < segs; i++) {
        const a = (i / segs) * Math.PI * 2;
        tRing.push(this.addVertex(cx + r1 * Math.cos(a), cy + r1 * Math.sin(a), cz + z1, 0, 0, 1, cr, cg, cb));
      }
      for (let i = 0; i < segs; i++) this.addTri(cTop, tRing[i], tRing[(i + 1) % segs]);
    }
  }

  // Swept circular tube along a 3D curve fn(t) -> [x,y,z] for t in [0, 1]
  // If closedLoop is true, connects end back to start; otherwise adds flat end caps.
  addTube(curveFn, radiusFn, uSegs, vSegs, col, { closedLoop = false } = {}) {
    const [cr, cg, cb] = col;
    const pts = [], tangs = [], rads = [];
    const eps = 1e-4;
    for (let i = 0; i <= uSegs; i++) {
      const t = i / uSegs;
      const p = curveFn(t);
      const t0 = closedLoop ? (t - eps + 1) % 1 : Math.max(0, t - eps);
      const t1 = closedLoop ? (t + eps) % 1 : Math.min(1, t + eps);
      const p0 = curveFn(t0), p1 = curveFn(t1);
      const tx = p1[0] - p0[0], ty = p1[1] - p0[1], tz = p1[2] - p0[2];
      const tl = Math.hypot(tx, ty, tz) || 1;
      pts.push(p);
      tangs.push([tx / tl, ty / tl, tz / tl]);
      rads.push(typeof radiusFn === 'function' ? radiusFn(t) : radiusFn);
    }
    // Parallel-transport frame along curve
    const N = [], B = [];
    {
      const t0 = tangs[0];
      const up = Math.abs(t0[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
      let nx = up[1] * t0[2] - up[2] * t0[1];
      let ny = up[2] * t0[0] - up[0] * t0[2];
      let nz = up[0] * t0[1] - up[1] * t0[0];
      const nl = Math.hypot(nx, ny, nz) || 1;
      nx /= nl; ny /= nl; nz /= nl;
      const bx = t0[1] * nz - t0[2] * ny;
      const by = t0[2] * nx - t0[0] * nz;
      const bz = t0[0] * ny - t0[1] * nx;
      N.push([nx, ny, nz]);
      B.push([bx, by, bz]);
    }
    for (let i = 1; i <= uSegs; i++) {
      const prevN = N[i - 1];
      const ti = tangs[i];
      // Project prevN orthogonal to ti
      const d = prevN[0] * ti[0] + prevN[1] * ti[1] + prevN[2] * ti[2];
      let nx = prevN[0] - d * ti[0], ny = prevN[1] - d * ti[1], nz = prevN[2] - d * ti[2];
      const nl = Math.hypot(nx, ny, nz) || 1;
      nx /= nl; ny /= nl; nz /= nl;
      const bx = ti[1] * nz - ti[2] * ny;
      const by = ti[2] * nx - ti[0] * nz;
      const bz = ti[0] * ny - ti[1] * nx;
      N.push([nx, ny, nz]);
      B.push([bx, by, bz]);
    }
    // If closedLoop, eliminate twist seam
    let twist = 0;
    if (closedLoop) {
      const n0 = N[0], nEnd = N[uSegs], b0 = B[0];
      const dotN = n0[0] * nEnd[0] + n0[1] * nEnd[1] + n0[2] * nEnd[2];
      const dotB = b0[0] * nEnd[0] + b0[1] * nEnd[1] + b0[2] * nEnd[2];
      twist = -Math.atan2(dotB, dotN);
    }
    const grid = [];
    for (let i = 0; i <= uSegs; i++) {
      const p = pts[i], n = N[i], b = B[i], rad = rads[i];
      const tw = twist * (i / uSegs);
      const row = [];
      for (let j = 0; j <= vSegs; j++) {
        const v = (j / vSegs) * Math.PI * 2 + tw;
        const cv = Math.cos(v), sv = Math.sin(v);
        const nx = n[0] * cv + b[0] * sv;
        const ny = n[1] * cv + b[1] * sv;
        const nz = n[2] * cv + b[2] * sv;
        row.push(this.addVertex(p[0] + rad * nx, p[1] + rad * ny, p[2] + rad * nz, nx, ny, nz, cr, cg, cb));
      }
      grid.push(row);
    }
    for (let i = 0; i < uSegs; i++) {
      for (let j = 0; j < vSegs; j++) {
        this.addQuad(grid[i][j], grid[i][j + 1], grid[i + 1][j + 1], grid[i + 1][j]);
      }
    }
    if (!closedLoop) {
      // Start cap (-tangs[0])
      const t0 = tangs[0], p0 = pts[0], n0 = N[0], b0 = B[0], r0 = rads[0];
      const c0 = this.addVertex(p0[0], p0[1], p0[2], -t0[0], -t0[1], -t0[2], cr, cg, cb);
      const rStart = [];
      for (let j = 0; j < vSegs; j++) {
        const v = (j / vSegs) * Math.PI * 2;
        const cv = Math.cos(v), sv = Math.sin(v);
        rStart.push(this.addVertex(
          p0[0] + r0 * (n0[0] * cv + b0[0] * sv),
          p0[1] + r0 * (n0[1] * cv + b0[1] * sv),
          p0[2] + r0 * (n0[2] * cv + b0[2] * sv),
          -t0[0], -t0[1], -t0[2], cr, cg, cb
        ));
      }
      for (let j = 0; j < vSegs; j++) this.addTri(c0, rStart[(j + 1) % vSegs], rStart[j]);

      // End cap (+tangs[uSegs])
      const t1 = tangs[uSegs], p1 = pts[uSegs], n1 = N[uSegs], b1 = B[uSegs], r1 = rads[uSegs];
      const c1 = this.addVertex(p1[0], p1[1], p1[2], t1[0], t1[1], t1[2], cr, cg, cb);
      const rEnd = [];
      for (let j = 0; j < vSegs; j++) {
        const v = (j / vSegs) * Math.PI * 2;
        const cv = Math.cos(v), sv = Math.sin(v);
        rEnd.push(this.addVertex(
          p1[0] + r1 * (n1[0] * cv + b1[0] * sv),
          p1[1] + r1 * (n1[1] * cv + b1[1] * sv),
          p1[2] + r1 * (n1[2] * cv + b1[2] * sv),
          t1[0], t1[1], t1[2], cr, cg, cb
        ));
      }
      for (let j = 0; j < vSegs; j++) this.addTri(c1, rEnd[j], rEnd[(j + 1) % vSegs]);
    }
  }

  // Rotated beveled bench slab (non-axis-aligned prism with rounded edges)
  addRotatedBeveledSlab(cx, cy, cz, len, wid, thk, yaw, col) {
    const [cr, cg, cb] = col;
    const ca = Math.cos(yaw), sa = Math.sin(yaw);
    const toWorld = (lx, ly, lz) => [cx + lx * ca - ly * sa, cy + lx * sa + ly * ca, cz + lz];
    const rotN = (nx, ny, nz) => [nx * ca - ny * sa, nx * sa + ny * ca, nz];
    // Rounded-rectangle cross-section in (ly, lz) extruded along lx in [-len/2, len/2]
    const hy = wid * 0.5, hz = thk * 0.5, bev = Math.min(0.03, hz * 0.45);
    const prof = [
      [-hy + bev, -hz], [hy - bev, -hz],
      [hy, -hz + bev], [hy, hz - bev],
      [hy - bev, hz], [-hy + bev, hz],
      [-hy, hz - bev], [-hy, -hz + bev],
    ];
    const nP = prof.length;
    const x0 = -len * 0.5, x1 = len * 0.5;
    for (let i = 0; i < nP; i++) {
      const p0 = prof[i], p1 = prof[(i + 1) % nP];
      const dy = p1[0] - p0[0], dz = p1[1] - p0[1];
      const nl = Math.hypot(dy, dz) || 1;
      const [nx, ny, nz] = rotN(0, dz / nl, -dy / nl);
      const v0 = this.addVertex(...toWorld(x0, p0[0], p0[1]), nx, ny, nz, cr, cg, cb);
      const v1 = this.addVertex(...toWorld(x1, p0[0], p0[1]), nx, ny, nz, cr, cg, cb);
      const v2 = this.addVertex(...toWorld(x1, p1[0], p1[1]), nx, ny, nz, cr, cg, cb);
      const v3 = this.addVertex(...toWorld(x0, p1[0], p1[1]), nx, ny, nz, cr, cg, cb);
      this.addQuad(v0, v3, v2, v1);
    }
    // End cap x0 (-X local)
    const [nx0, ny0, nz0] = rotN(-1, 0, 0);
    const c0 = this.addVertex(...toWorld(x0, 0, 0), nx0, ny0, nz0, cr, cg, cb);
    const r0 = prof.map(([y, z]) => this.addVertex(...toWorld(x0, y, z), nx0, ny0, nz0, cr, cg, cb));
    for (let i = 0; i < nP; i++) this.addTri(c0, r0[(i + 1) % nP], r0[i]);
    // End cap x1 (+X local)
    const [nx1, ny1, nz1] = rotN(1, 0, 0);
    const c1 = this.addVertex(...toWorld(x1, 0, 0), nx1, ny1, nz1, cr, cg, cb);
    const r1 = prof.map(([y, z]) => this.addVertex(...toWorld(x1, y, z), nx1, ny1, nz1, cr, cg, cb));
    for (let i = 0; i < nP; i++) this.addTri(c1, r1[i], r1[(i + 1) % nP]);
  }
}

export function buildAtriumGeometryData() {
  const b = new MeshBuilder();

  // Warm architectural palette (linear RGB)
  const stoneFloor = [0.46, 0.42, 0.36];
  const stoneCurb  = [0.54, 0.49, 0.41];
  const plaster    = [0.64, 0.56, 0.45];
  const marble     = [0.72, 0.68, 0.60];
  const travertine = [0.58, 0.52, 0.43];
  const terracotta = [0.56, 0.24, 0.16];
  const glazedClay = [0.62, 0.33, 0.21];
  const bronze     = [0.48, 0.34, 0.19];
  const alabaster  = [0.76, 0.72, 0.66];
  const benchStone = [0.38, 0.36, 0.33];

  // 1. Circular rotunda stone floor + stepped outer curb ring
  b.addCylinder(0, 0, -0.35, 0.0, 5.65, 5.65, 36, stoneFloor);
  b.addArcWall(5.12, 5.65, 0.0, 0.16, 0, Math.PI * 2, 36, stoneCurb);

  // 2. Curved apse wall around the back & sides (-22 deg to 202 deg, open toward -Y)
  const apseA0 = -22 * Math.PI / 180;
  const apseA1 = 202 * Math.PI / 180;
  b.addArcWall(5.05, 5.55, 0.0, 4.55, apseA0, apseA1, 28, plaster);
  // Base plinth molding & overhanging cornice along apse wall
  b.addArcWall(4.86, 5.08, 0.0, 0.26, apseA0, apseA1, 28, stoneCurb);
  b.addArcWall(4.68, 5.62, 4.42, 4.82, apseA0, apseA1, 28, marble);

  // 3. Eight fluted cylindrical columns in a ring (R = 3.75 m)
  const colR = 3.75;
  const colAngles = [];
  for (let k = 0; k < 8; k++) {
    const ang = (22.5 + k * 45) * Math.PI / 180;
    colAngles.push(ang);
    const cx = colR * Math.cos(ang), cy = colR * Math.sin(ang);
    // Circular plinth
    b.addCylinder(cx, cy, 0.0, 0.12, 0.35, 0.33, 12, stoneCurb);
    // Torus base ring
    b.addTorus(cx, cy, 0.16, 0.25, 0.065, 12, 6, marble);
    // Fluted column shaft (10 flutes)
    b.addCylinder(cx, cy, 0.18, 3.28, 0.24, 0.21, 20, marble, { flutes: 10, fluteDepth: 0.022, angleOffset: ang });
    // Torus echinus capital
    b.addTorus(cx, cy, 3.31, 0.23, 0.07, 12, 6, marble);
    // Circular abacus cap
    b.addCylinder(cx, cy, 3.35, 3.52, 0.31, 0.35, 12, stoneCurb);
  }

  // 4. Eight curved vaulted arches linking adjacent columns + upper rotunda entablature ring
  for (let k = 0; k < 8; k++) {
    const a0 = colAngles[k];
    const a1 = colAngles[(k + 1) % 8];
    let da = a1 - a0;
    if (da < 0) da += Math.PI * 2;
    const archCurve = (t) => {
      const a = a0 + t * da;
      const z = 3.42 + 1.06 * Math.sin(t * Math.PI);
      return [colR * Math.cos(a), colR * Math.sin(a), z];
    };
    b.addTube(archCurve, 0.135, 12, 6, marble, { closedLoop: false });
  }
  // Circular entablature ring crowning the arches
  b.addArcWall(3.52, 4.02, 4.46, 4.86, 0, Math.PI * 2, 32, travertine);
  b.addTorus(0, 0, 4.48, 3.54, 0.055, 32, 6, marble);

  // 5. Thin curved balustrade railings in the 5 back/side bays
  const balusterBays = [0, 1, 2, 3, 7];
  for (const k of balusterBays) {
    const a0 = colAngles[k] + 0.055;
    const a1 = colAngles[k] + (45 * Math.PI / 180) - 0.055;
    const railCurve = (z) => (t) => {
      const a = a0 + t * (a1 - a0);
      return [colR * Math.cos(a), colR * Math.sin(a), z];
    };
    // Bottom & top thin curved rails
    b.addTube(railCurve(0.16), 0.042, 8, 6, stoneCurb, { closedLoop: false });
    b.addTube(railCurve(0.92), 0.048, 8, 6, marble, { closedLoop: false });
    // 5 slender cylindrical balusters per bay
    for (let j = 1; j <= 5; j++) {
      const a = a0 + (j / 6) * (a1 - a0);
      b.addCylinder(colR * Math.cos(a), colR * Math.sin(a), 0.14, 0.92, 0.032, 0.026, 6, marble);
    }
  }

  // 6. Central 3-tier circular pedestal
  b.addCylinder(0, 0, 0.0, 0.28, 1.65, 1.62, 24, travertine);
  b.addTorus(0, 0, 0.28, 1.61, 0.038, 24, 6, marble);
  b.addCylinder(0, 0, 0.28, 0.68, 1.16, 1.13, 22, stoneCurb);
  b.addTorus(0, 0, 0.68, 1.12, 0.034, 22, 6, marble);
  b.addCylinder(0, 0, 0.68, 1.14, 0.66, 0.63, 20, marble);
  b.addTorus(0, 0, 1.14, 0.62, 0.03, 20, 6, travertine);

  // 7. Central Torus Knot sculpture (p=2, q=3) resting on top pedestal tier (z = 1.14)
  const knotCenterZ = 1.66;
  const knotCurve = (t) => {
    const phi = t * Math.PI * 2;
    const p = 2, q = 3;
    const r = 0.27 + 0.13 * Math.cos(q * phi);
    const x = r * Math.cos(p * phi);
    const y = r * Math.sin(p * phi);
    const z = 0.22 * Math.sin(q * phi);
    const tilt = 0.35;
    const ct = Math.cos(tilt), st = Math.sin(tilt);
    return [x, y * ct - z * st, knotCenterZ + y * st + z * ct];
  };
  b.addTube(knotCurve, 0.088, 44, 8, bronze, { closedLoop: true });
  b.addCylinder(0, 0, 1.14, 1.38, 0.14, 0.08, 12, bronze);

  // 8. Closed sculptural Teapot on Tier 2 at (-0.82, -0.22, 0.68), rotated 35 deg
  {
    const tx = -0.82, ty = -0.22, tz = 0.68;
    const rot = 0.62;
    const cr = Math.cos(rot), sr = Math.sin(rot);
    const toW = (lx, ly, lz) => [tx + lx * cr - ly * sr, ty + lx * sr + ly * cr, tz + lz];
    const potProfile = [
      [0.14, 0.00], [0.22, 0.06], [0.26, 0.15], [0.25, 0.24],
      [0.19, 0.31], [0.15, 0.34], [0.17, 0.36], [0.13, 0.39],
      [0.05, 0.41], [0.065, 0.44], [0.025, 0.465], [0.0, 0.47],
    ];
    b.addLathe(tx, ty, tz, potProfile, 14, terracotta);
    b.addTube(
      (t) => toW(0.21 + 0.22 * t, 0, 0.14 + 0.20 * Math.pow(t, 1.4)),
      (t) => 0.048 * (1 - 0.55 * t),
      8, 6, terracotta, { closedLoop: false }
    );
    b.addTube(
      (t) => {
        const a = -1.1 + t * 2.25;
        return toW(-0.22 - 0.15 * Math.cos(a), 0, 0.21 + 0.11 * Math.sin(a));
      },
      0.026, 10, 6, terracotta, { closedLoop: false }
    );
  }

  // 9. Lathed ceramic amphorae / vases
  const vaseProfile = [
    [0.11, 0.00], [0.14, 0.04], [0.10, 0.09], [0.20, 0.24],
    [0.23, 0.38], [0.18, 0.50], [0.09, 0.58], [0.12, 0.66],
    [0.15, 0.70], [0.12, 0.72], [0.08, 0.68], [0.0, 0.66],
  ];
  b.addLathe(0.84, 0.22, 0.68, vaseProfile, 14, glazedClay);
  const tallVaseProfile = vaseProfile.map(([r, z]) => [r * 1.25, z * 1.35]);
  b.addLathe(-2.45, -1.65, 0.0, tallVaseProfile, 16, terracotta);

  // 10. Polished alabaster spheres in tight contacts on pedestal tiers & floor
  b.addSphere(-0.42, -1.28, 0.28 + 0.19, 0.19, 14, 10, alabaster);
  b.addSphere(0.12, -1.36, 0.28 + 0.13, 0.13, 12, 8, alabaster);
  b.addSphere(0.76, -0.58, 0.68 + 0.14, 0.14, 12, 8, alabaster);
  b.addSphere(1.95, -1.35, 0.24, 0.24, 14, 10, alabaster);

  // 11. Four angled stone benches at non-90° angles (45°, 135°, 225°, 315° radial placement)
  const benchAngles = [45, 135, 225, 315].map((d) => d * Math.PI / 180);
  for (const ang of benchAngles) {
    const r = 2.68;
    const cx = r * Math.cos(ang), cy = r * Math.sin(ang);
    const yaw = ang + Math.PI / 2;
    const ca = Math.cos(yaw), sa = Math.sin(yaw);
    b.addRotatedBeveledSlab(cx, cy, 0.44, 1.36, 0.44, 0.11, yaw, benchStone);
    for (const side of [-0.46, 0.46]) {
      const lx = cx + side * ca, ly = cy + side * sa;
      b.addCylinder(lx, ly, 0.0, 0.39, 0.14, 0.11, 10, stoneCurb);
      b.addTorus(lx, ly, 0.05, 0.135, 0.032, 10, 6, marble);
    }
  }

  return {
    positions: new Float32Array(b.pos),
    normals: new Float32Array(b.nor),
    colors: new Float32Array(b.col),
    indices: new Uint32Array(b.idx),
    triangleCount: b.idx.length / 3,
    vertexCount: b.pos.length / 3,
  };
}

export function buildAtriumThreeGeometry(T) {
  const data = buildAtriumGeometryData();
  const geo = new T.BufferGeometry();
  geo.setAttribute('position', new T.BufferAttribute(data.positions, 3));
  geo.setAttribute('normal', new T.BufferAttribute(data.normals, 3));
  geo.setAttribute('surfAlbedo', new T.BufferAttribute(data.colors, 3));
  geo.setIndex(new T.BufferAttribute(data.indices, 1));
  geo.computeBoundingBox();
  geo.computeBoundingSphere();
  return { geometry: geo, data };
}
