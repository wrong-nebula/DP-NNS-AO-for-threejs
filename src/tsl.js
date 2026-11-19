// Fast-SDF-AO Three.js r187+ WebGPU / TSL Node.
// Plugs directly into material.aoNode with zero pre-pass and zero post-processing:
//
//   import { loadSDF } from 'fast-sdf-ao';
//   import { createSDFAONode } from 'fast-sdf-ao/tsl';
//   material.aoNode = createSDFAONode({ field, min: [-6.5, -6.5, -0.5], span: [13, 13, 6.5] });

import {
  Fn,
  clamp,
  float,
  max,
  mix,
  normalWorld,
  positionWorld,
  texture3D,
  vec3,
} from 'three/tsl';

function toVec3(v, name) {
  if (Array.isArray(v) && v.length === 3) return vec3(v[0], v[1], v[2]);
  if (v && typeof v.x === 'number' && typeof v.y === 'number' && typeof v.z === 'number') {
    return vec3(v.x, v.y, v.z);
  }
  throw new Error(`Invalid ${name}: expected [x, y, z] or THREE.Vector3`);
}

/**
 * Create a Three.js TSL node computing generic 4-tap SDF ambient occlusion in
 * world space. Assign directly to `material.aoNode`.
 */
export function createSDFAONode({
  field,
  min,
  span,
  radius = 1.12,
  intensity = 1.0,
} = {}) {
  if (!field) throw new Error('createSDFAONode requires a Data3DTexture { field }');
  const sdfMin = toVec3(min, 'min');
  const spanArr = Array.isArray(span) ? span : [span.x, span.y, span.z];
  const sdfInvSpan = vec3(1 / spanArr[0], 1 / spanArr[1], 1 / spanArr[2]);
  const r0 = float(0.142857 * radius);
  const r1 = float(0.428571 * radius);
  const r2 = float(0.714286 * radius);
  const r3 = float(1.0 * radius);
  const strength = float(intensity);

  const sampleStatic = (pos) => {
    const uv = pos.sub(sdfMin).mul(sdfInvSpan);
    const inBounds = uv.x.greaterThanEqual(0.0)
      .and(uv.y.greaterThanEqual(0.0))
      .and(uv.z.greaterThanEqual(0.0))
      .and(uv.x.lessThanEqual(1.0))
      .and(uv.y.lessThanEqual(1.0))
      .and(uv.z.lessThanEqual(1.0));
    const sampled = texture3D(field, clamp(uv, vec3(0.0), vec3(1.0))).r;
    return inBounds.select(sampled, float(1e4));
  };

  const sdfAOFn = Fn(() => {
    const p = positionWorld;
    const n = normalWorld.normalize();

    const d0 = sampleStatic(p.add(n.mul(r0)));
    const d1 = sampleStatic(p.add(n.mul(r1)));
    const d2 = sampleStatic(p.add(n.mul(r2)));
    const d3 = sampleStatic(p.add(n.mul(r3)));

    const o0 = clamp(r0.sub(d0).sub(0.045).div(r0), 0.0, 1.0);
    const o1 = clamp(r1.sub(d1).sub(0.045).div(r1), 0.0, 1.0);
    const o2 = clamp(r2.sub(d2).sub(0.045).div(r2), 0.0, 1.0);
    const o3 = clamp(r3.sub(d3).sub(0.045).div(r3), 0.0, 1.0);

    const weighted = o0.mul(1.0)
      .add(o1.mul(0.816667))
      .add(o2.mul(0.633333))
      .add(o3.mul(0.45));

    const ao = clamp(float(1.0).sub(weighted.mul(0.92 / 2.9)), 0.0, 1.0);
    return clamp(mix(float(1.0), ao, strength), 0.0, 1.0);
  });

  return sdfAOFn();
}

/**
 * Jimenez 2016 Multi-Bounce Colored AO TSL node.
 */
export const createMultiBounceAONode = Fn(([ao, albedo]) => {
  const a = albedo.mul(2.0404).sub(0.3324);
  const b = albedo.mul(-4.7951).add(0.6417);
  const c = albedo.mul(2.7552).add(0.6903);
  const poly = a.mul(ao).add(b).mul(ao).add(c).mul(ao);
  return max(vec3(ao), poly);
});
