// trace_tris: any-hit ray tests against a triangle soup, for the ray-traced AO reference.
//   trace_tris tris.f32 rays.f32 hits.u8
// tris.f32: 9 floats per triangle (world space). rays.f32: 7 floats per ray (origin, unit
// direction, tmax). hits.u8: 1 where the ray hits a triangle at 0 <= t <= tmax.
// Median-split BVH, Moller-Trumbore with double-sided hits, all cores.
// Build: c++ -O3 -std=c++17 -pthread tools/trace_tris.cpp -o trace_tris
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

struct V { float x, y, z; };
static V sub(V a, V b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static V cross(V a, V b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
static float dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
struct Tri { V a, b, c; };
struct Node { float lo[3], hi[3]; int left = -1, right = -1, start = 0, count = 0; };

template <class T> static std::vector<T> load(const std::string& p) {
    std::ifstream in(p, std::ios::binary | std::ios::ate); if (!in) throw std::runtime_error("cannot read " + p);
    size_t n = size_t(in.tellg()); in.seekg(0); std::vector<T> v(n / sizeof(T)); in.read((char*)v.data(), n); return v;
}

struct BVH {
    const std::vector<Tri>& t; std::vector<uint32_t> order; std::vector<Node> nodes;
    explicit BVH(const std::vector<Tri>& tris) : t(tris) {
        order.resize(t.size()); std::iota(order.begin(), order.end(), 0); build(0, int(t.size()));
    }
    float cen(uint32_t i, int ax) const { const Tri& x = t[i]; const float* a = &x.a.x; const float* b = &x.b.x; const float* c = &x.c.x; return (a[ax] + b[ax] + c[ax]) / 3; }
    int build(int s, int e) {
        Node n; for (int k = 0; k < 3; ++k) { n.lo[k] = INFINITY; n.hi[k] = -INFINITY; }
        for (int i = s; i < e; ++i) for (const V* p : {&t[order[i]].a, &t[order[i]].b, &t[order[i]].c}) { const float* q = &p->x; for (int k = 0; k < 3; ++k) { n.lo[k] = std::min(n.lo[k], q[k]); n.hi[k] = std::max(n.hi[k], q[k]); } }
        const int id = int(nodes.size()); nodes.push_back(n);
        if (e - s <= 4) { nodes[id].start = s; nodes[id].count = e - s; return id; }
        int ax = 0; float ext = -1; for (int k = 0; k < 3; ++k) if (n.hi[k] - n.lo[k] > ext) { ext = n.hi[k] - n.lo[k]; ax = k; }
        const int m = (s + e) / 2;
        std::nth_element(order.begin() + s, order.begin() + m, order.begin() + e, [&](uint32_t a, uint32_t b) { return cen(a, ax) < cen(b, ax); });
        const int l = build(s, m), r = build(m, e); nodes[id].left = l; nodes[id].right = r; return id;
    }
    bool box(const Node& n, V o, V inv, float tmax) const {
        float t0 = 0, t1 = tmax; const float* oo = &o.x; const float* ii = &inv.x;
        for (int k = 0; k < 3; ++k) { float a = (n.lo[k] - oo[k]) * ii[k], b = (n.hi[k] - oo[k]) * ii[k]; if (a > b) std::swap(a, b); t0 = std::max(t0, a); t1 = std::min(t1, b); }
        return t0 <= t1;
    }
    bool any(V o, V d, float tmax) const {
        V inv{1 / d.x, 1 / d.y, 1 / d.z}; int stack[128], sp = 0; stack[sp++] = 0;
        while (sp) {
            const Node& n = nodes[stack[--sp]];
            if (!box(n, o, inv, tmax)) continue;
            if (n.count) {
                for (int i = 0; i < n.count; ++i) {
                    const Tri& x = t[order[n.start + i]]; V e1 = sub(x.b, x.a), e2 = sub(x.c, x.a), p = cross(d, e2); float det = dot(e1, p);
                    if (std::fabs(det) < 1e-12f) continue; float id = 1 / det; V s = sub(o, x.a); float u = dot(s, p) * id; if (u < 0 || u > 1) continue;
                    V q = cross(s, e1); float v = dot(d, q) * id; if (v < 0 || u + v > 1) continue; float tt = dot(e2, q) * id;
                    if (tt >= 0 && tt <= tmax) return true;
                }
                continue;
            }
            stack[sp++] = n.left; stack[sp++] = n.right;
        }
        return false;
    }
};

int main(int argc, char** argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: trace_tris tris.f32 rays.f32 hits.u8\n"); return 2; }
    auto tris = load<Tri>(argv[1]); auto rays = load<float>(argv[2]);
    const size_t n = rays.size() / 7; std::vector<uint8_t> hits(n, 0);
    BVH bvh(tris); std::atomic<size_t> next{0};
    auto work = [&] { for (size_t b; (b = next.fetch_add(4096)) < n;) for (size_t i = b; i < std::min(n, b + 4096); ++i) {
        const float* r = &rays[7 * i]; hits[i] = bvh.any({r[0], r[1], r[2]}, {r[3], r[4], r[5]}, r[6]); } };
    std::vector<std::thread> pool; for (unsigned k = 0; k < std::max(1u, std::thread::hardware_concurrency()); ++k) pool.emplace_back(work);
    for (auto& p : pool) p.join();
    std::ofstream(argv[3], std::ios::binary).write((const char*)hits.data(), hits.size());
    std::printf("%zu triangles, %zu rays, %zu hits\n", tris.size(), n, size_t(std::count(hits.begin(), hits.end(), 1)));
}
