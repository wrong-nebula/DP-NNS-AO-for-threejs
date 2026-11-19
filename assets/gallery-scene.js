// Procedural Sunlit Color-Bleeding Gallery scene (zero external dependencies).
// Cornell-inspired architectural gallery with saturated terracotta crimson left wall,
// Mediterranean cobalt/teal right wall, warm matte white/alabaster walls, ceiling
// skylight opening, golden ochre floor inlay, stepped central pedestal with smooth
// alabaster sphere & curved torus knot, angled architectural monolith, back curved
// archway alcove, and right gallery bench with amphora & sphere.
// Shared identically between tools/gen_gallery.mjs (SDF bake & 512-ray GI reference)
// and index.html (Three.js WebGL2 runtime).

export const GALLERY_SDF_MIN = [-4.0, -3.8, -0.45];
export const GALLERY_SDF_SPAN = [8.0, 8.0, 5.333333333333333];
export const GALLERY_SDF_DIMS = [96, 96, 64];

export const GALLERY_CAMERA = {
  position: [0.0, -5.95, 2.15],
  target: [0.0, 0.75, 1.55],
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

  // Axis-aligned closed box [x0, x1] x [y0, y1] x [z0, z1], 100% outward-wound
  addBox(x0, x1, y0, y1, z0, z1, col) {
    const [cr, cg, cb] = col;
    const face = (p0, p1, p2, p3, nx, ny, nz) => {
      const a = this.addVertex(p0[0], p0[1], p0[2], nx, ny, nz, cr, cg, cb);
      const b = this.addVertex(p1[0], p1[1], p1[2], nx, ny, nz, cr, cg, cb);
      const c = this.addVertex(p2[0], p2[1], p2[2], nx, ny, nz, cr, cg, cb);
      const d = this.addVertex(p3[0], p3[1], p3[2], nx, ny, nz, cr, cg, cb);
      this.addQuad(a, b, c, d);
    };
    // +Z top
    face([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1], 0, 0, 1);
    // -Z bottom
    face([x0, y1, z0], [x1, y1, z0], [x1, y0, z0], [x0, y0, z0], 0, 0, -1);
    // +Y back
    face([x1, y1, z0], [x0, y1, z0], [x0, y1, z1], [x1, y1, z1], 0, 1, 0);
    // -Y front
    face([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1], 0, -1, 0);
    // +X right
    face([x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1], 1, 0, 0);
    // -X left
    face([x0, y1, z0], [x0, y0, z0], [x0, y0, z1], [x0, y1, z1], -1, 0, 0);
  }

  // Closed cylinder / frustum along Z from z0 to z1
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
      const aP = a + eps, aM = a - eps;
      const rP = radAt(aP, 0.5 * (r0 + r1)), rM = radAt(aM, 0.5 * (r0 + r1));
      const tx = (rP * Math.cos(aP) - rM * Math.cos(aM)) / (2 * eps);
      const ty = (rP * Math.sin(aP) - rM * Math.sin(aM)) / (2 * eps);
      const dr = rad1 - rad0;
      const tzX = dr * ca, tzY = dr * sa, tzZ = dz;
      const nx = ty * tzZ;
      const ny = -tx * tzZ;
      const nz = tx * tzY - ty * tzX;
      ring0.push(this.addVertex(cx + rad0 * ca, cy + rad0 * sa, z0, nx, ny, nz, cr, cg, cb));
      ring1.push(this.addVertex(cx + rad1 * ca, cy + rad1 * sa, z1, nx, ny, nz, cr, cg, cb));
    }
    for (let i = 0; i < segs; i++) {
      this.addQuad(ring0[i], ring0[i + 1], ring1[i + 1], ring1[i]);
    }
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

  // Vertical bevelled monolith / slab rotated around Z by yaw, spanning z in [z0, z1]
  addVerticalBeveledSlab(cx, cy, z0, z1, wid, dep, yaw, col) {
    const [cr, cg, cb] = col;
    const ca = Math.cos(yaw), sa = Math.sin(yaw);
    const toW = (lx, ly, lz) => [cx + lx * ca - ly * sa, cy + lx * sa + ly * ca, lz];
    const rotN = (nx, ny, nz) => [nx * ca - ny * sa, nx * sa + ny * ca, nz];
    const hx = wid * 0.5, hy = dep * 0.5, bev = Math.min(0.04, hx * 0.2, hy * 0.2);
    // Counter-clockwise profile in XY so outward normal is (dy, -dx, 0)
    const prof = [
      [-hx + bev, -hy], [hx - bev, -hy],
      [hx, -hy + bev], [hx, hy - bev],
      [hx - bev, hy], [-hx + bev, hy],
      [-hx, hy - bev], [-hx, -hy + bev],
    ];
    const nP = prof.length;
    for (let i = 0; i < nP; i++) {
      const p0 = prof[i], p1 = prof[(i + 1) % nP];
      const dx = p1[0] - p0[0], dy = p1[1] - p0[1];
      const nl = Math.hypot(dx, dy) || 1;
      const [nx, ny, nz] = rotN(dy / nl, -dx / nl, 0);
      const v0 = this.addVertex(...toW(p0[0], p0[1], z0), nx, ny, nz, cr, cg, cb);
      const v1 = this.addVertex(...toW(p1[0], p1[1], z0), nx, ny, nz, cr, cg, cb);
      const v2 = this.addVertex(...toW(p1[0], p1[1], z1), nx, ny, nz, cr, cg, cb);
      const v3 = this.addVertex(...toW(p0[0], p0[1], z1), nx, ny, nz, cr, cg, cb);
      this.addQuad(v0, v1, v2, v3);
    }
    // Bottom cap (-Z)
    const cBot = this.addVertex(cx, cy, z0, 0, 0, -1, cr, cg, cb);
    const rBot = prof.map(([x, y]) => this.addVertex(...toW(x, y, z0), 0, 0, -1, cr, cg, cb));
    for (let i = 0; i < nP; i++) this.addTri(cBot, rBot[(i + 1) % nP], rBot[i]);
    // Top cap (+Z)
    const cTop = this.addVertex(cx, cy, z1, 0, 0, 1, cr, cg, cb);
    const rTop = prof.map(([x, y]) => this.addVertex(...toW(x, y, z1), 0, 0, 1, cr, cg, cb));
    for (let i = 0; i < nP; i++) this.addTri(cTop, rTop[i], rTop[(i + 1) % nP]);
  }

  // Horizontal bevelled bench slab rotated around Z by yaw
  addHorizontalBeveledSlab(cx, cy, cz, len, wid, thk, yaw, col) {
    const [cr, cg, cb] = col;
    const ca = Math.cos(yaw), sa = Math.sin(yaw);
    const toW = (lx, ly, lz) => [cx + lx * ca - ly * sa, cy + lx * sa + ly * ca, cz + lz];
    const rotN = (nx, ny, nz) => [nx * ca - ny * sa, nx * sa + ny * ca, nz];
    const hy = wid * 0.5, hz = thk * 0.5, bev = Math.min(0.025, hz * 0.4);
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
      const v0 = this.addVertex(...toW(x0, p0[0], p0[1]), nx, ny, nz, cr, cg, cb);
      const v1 = this.addVertex(...toW(x1, p0[0], p0[1]), nx, ny, nz, cr, cg, cb);
      const v2 = this.addVertex(...toW(x1, p1[0], p1[1]), nx, ny, nz, cr, cg, cb);
      const v3 = this.addVertex(...toW(x0, p1[0], p1[1]), nx, ny, nz, cr, cg, cb);
      this.addQuad(v0, v3, v2, v1);
    }
    const [nx0, ny0, nz0] = rotN(-1, 0, 0);
    const c0 = this.addVertex(...toW(x0, 0, 0), nx0, ny0, nz0, cr, cg, cb);
    const r0 = prof.map(([y, z]) => this.addVertex(...toW(x0, y, z), nx0, ny0, nz0, cr, cg, cb));
    for (let i = 0; i < nP; i++) this.addTri(c0, r0[(i + 1) % nP], r0[i]);
    const [nx1, ny1, nz1] = rotN(1, 0, 0);
    const c1 = this.addVertex(...toW(x1, 0, 0), nx1, ny1, nz1, cr, cg, cb);
    const r1 = prof.map(([y, z]) => this.addVertex(...toW(x1, y, z), nx1, ny1, nz1, cr, cg, cb));
    for (let i = 0; i < nP; i++) this.addTri(c1, r1[i], r1[(i + 1) % nP]);
  }
}

export function buildGalleryGeometryData() {
  const b = new MeshBuilder();

  // Iconic Cornell-inspired Sunlit Gallery palette (linear RGB)
  const alabaster  = [0.84, 0.82, 0.78]; // Floor, back wall, ceiling, hero sculptures
  const warmWhite  = [0.86, 0.84, 0.80]; // Sculptures & arch trim
  const crimson    = [0.80, 0.13, 0.09]; // Saturated terracotta crimson left wall
  const cobaltTeal = [0.09, 0.48, 0.64]; // Rich Mediterranean cobalt/teal right wall
  const goldenRug  = [0.82, 0.58, 0.12]; // Bright golden ochre floor inlay under skylight

  // 1. Thick watertight room shell (interior: X [-3.2, 3.2], Y [-3.0, 3.2], Z [0.0, 4.4])
  // Open on the front (-Y) camera side.
  // Floor
  b.addBox(-3.60, 3.60, -3.20, 3.60, -0.35, 0.0, alabaster);
  // Left wall (terracotta crimson)
  b.addBox(-3.60, -3.20, -3.20, 3.60, 0.0, 4.40, crimson);
  // Right wall (Mediterranean cobalt/teal)
  b.addBox(3.20, 3.60, -3.20, 3.60, 0.0, 4.40, cobaltTeal);
  // Back wall (warm matte white / alabaster)
  b.addBox(-3.60, 3.60, 3.20, 3.60, 0.0, 4.40, alabaster);

  // 2. Ceiling with generous rectangular skylight opening: X [-2.65, 2.65], Y [-1.60, 1.90]
  // Front ceiling beam
  b.addBox(-3.60, 3.60, -3.20, -1.60, 4.40, 4.82, alabaster);
  // Back ceiling beam (casts crisp shadow over the back wall & archway alcove!)
  b.addBox(-3.60, 3.60, 1.90, 3.60, 4.40, 4.82, alabaster);
  // Left ceiling beam
  b.addBox(-3.60, -2.65, -1.60, 1.90, 4.40, 4.82, alabaster);
  // Right ceiling beam
  b.addBox(2.65, 3.60, -1.60, 1.90, 4.40, 4.82, alabaster);
  // Subtle ceiling coffer trim framing the skylight
  b.addBox(-2.73, 2.73, -1.68, -1.60, 4.30, 4.40, warmWhite);
  b.addBox(-2.73, 2.73, 1.90, 1.98, 4.30, 4.40, warmWhite);
  b.addBox(-2.73, -2.65, -1.60, 1.90, 4.30, 4.40, warmWhite);
  b.addBox(2.65, 2.73, -1.60, 1.90, 4.30, 4.40, warmWhite);

  // 3. Golden ochre floor rug / inlay centered under the skylight beam
  b.addBox(-2.15, 2.15, -1.65, 1.85, 0.0, 0.025, goldenRug);

  // 4. Central stepped white circular pedestal (at x=0, y=0.10, resting on rug z=0.025)
  b.addCylinder(0.0, 0.10, 0.025, 0.28, 1.24, 1.21, 32, warmWhite);
  b.addTorus(0.0, 0.10, 0.28, 1.20, 0.032, 32, 8, warmWhite);
  b.addCylinder(0.0, 0.10, 0.28, 0.68, 0.84, 0.81, 28, warmWhite);
  b.addTorus(0.0, 0.10, 0.68, 0.80, 0.028, 28, 8, warmWhite);

  // 5. Large smooth alabaster sphere on the pedestal (right-front)
  b.addSphere(0.36, -0.16, 0.68 + 0.40, 0.40, 36, 22, warmWhite);

  // 6. White curved torus knot sculpture (p=2, q=3) on the pedestal (left-center)
  const knotCx = -0.34, knotCy = 0.26, knotCz = 1.44;
  const knotCurve = (t) => {
    const phi = t * Math.PI * 2;
    const p = 2, q = 3;
    const r = 0.25 + 0.12 * Math.cos(q * phi);
    const x = r * Math.cos(p * phi);
    const y = r * Math.sin(p * phi);
    const z = 0.20 * Math.sin(q * phi);
    const tilt = 0.42;
    const ct = Math.cos(tilt), st = Math.sin(tilt);
    return [knotCx + x, knotCy + y * ct - z * st, knotCz + y * st + z * ct];
  };
  b.addTube(knotCurve, 0.088, 60, 12, warmWhite, { closedLoop: true });
  b.addCylinder(knotCx, knotCy, 0.68, 1.16, 0.12, 0.075, 16, warmWhite);

  // 7. Tall angled white architectural monolith / slab on the left-back (catches crimson bounce!)
  b.addVerticalBeveledSlab(-2.05, 1.35, 0.0, 2.75, 0.96, 0.42, 0.44, alabaster);
  // Smaller white companion cube / plinth beside it
  b.addVerticalBeveledSlab(-1.82, -0.55, 0.0, 0.92, 0.64, 0.64, -0.30, warmWhite);
  b.addSphere(-1.82, -0.55, 0.92 + 0.22, 0.22, 24, 16, warmWhite);

  // 8. White curved archway / alcove against the back wall (shadowed from direct sun!)
  // Two fluted white pilasters + semicircular curved arch + recessed cornice shelf
  for (const sx of [-1.22, 1.22]) {
    b.addCylinder(sx, 2.68, 0.0, 0.14, 0.30, 0.28, 16, alabaster);
    b.addTorus(sx, 2.68, 0.17, 0.22, 0.05, 16, 8, warmWhite);
    b.addCylinder(sx, 2.68, 0.18, 2.22, 0.21, 0.19, 24, warmWhite, { flutes: 10, fluteDepth: 0.018 });
    b.addTorus(sx, 2.68, 2.25, 0.21, 0.055, 16, 8, warmWhite);
    b.addCylinder(sx, 2.68, 2.28, 2.42, 0.27, 0.30, 16, alabaster);
  }
  // Semicircular arch spanning x = -1.22 to +1.22 at y = 2.68, springing from z = 2.38
  b.addTube(
    (t) => {
      const a = Math.PI * (1 - t);
      return [1.22 * Math.cos(a), 2.68, 2.38 + 1.18 * Math.sin(a)];
    },
    0.16, 28, 12, warmWhite, { closedLoop: false }
  );
  // Subtle back-wall gallery plinth inside the shadowed alcove with a white amphora
  b.addBox(-0.62, 0.62, 2.55, 3.20, 0.0, 0.72, alabaster);
  const amphoraProfile = [
    [0.13, 0.00], [0.16, 0.05], [0.11, 0.12], [0.24, 0.30],
    [0.27, 0.48], [0.20, 0.64], [0.10, 0.74], [0.13, 0.84],
    [0.17, 0.90], [0.13, 0.92], [0.08, 0.86], [0.0, 0.84],
  ];
  b.addLathe(0.0, 2.86, 0.72, amphoraProfile, 24, warmWhite);

  // 9. Right-side low white gallery bench + matte white spheres (catches rich teal bounce!)
  b.addHorizontalBeveledSlab(2.15, 0.55, 0.52, 1.75, 0.54, 0.12, Math.PI * 0.5 + 0.12, warmWhite);
  for (const dy of [-0.54, 0.54]) {
    const lx = 2.15 - dy * Math.sin(0.12);
    const ly = 0.55 + dy * Math.cos(0.12);
    b.addCylinder(lx, ly, 0.0, 0.47, 0.14, 0.12, 14, alabaster);
  }
  b.addSphere(2.18, 0.25, 0.58 + 0.21, 0.21, 24, 16, warmWhite);
  b.addSphere(2.08, 0.95, 0.58 + 0.15, 0.15, 20, 14, warmWhite);

  return {
    positions: new Float32Array(b.pos),
    normals: new Float32Array(b.nor),
    colors: new Float32Array(b.col),
    indices: new Uint32Array(b.idx),
    triangleCount: b.idx.length / 3,
    vertexCount: b.pos.length / 3,
  };
}

export function buildGalleryThreeGeometry(T) {
  const data = buildGalleryGeometryData();
  const geo = new T.BufferGeometry();
  geo.setAttribute('position', new T.BufferAttribute(data.positions, 3));
  geo.setAttribute('normal', new T.BufferAttribute(data.normals, 3));
  geo.setAttribute('surfAlbedo', new T.BufferAttribute(data.colors, 3));
  geo.setIndex(new T.BufferAttribute(data.indices, 1));
  geo.computeBoundingBox();
  geo.computeBoundingSphere();
  return { geometry: geo, data };
}
