// Point-cloud SDF bake benchmark: DP-NNS (arXiv 2611.15320) vs a kd-tree (nanoflann) vs a BVH over points.
//
// The scene's surfaces are sampled into an oriented point cloud (position + normal), with points hidden
// under other boxes removed, as a scan would see the scene. Every method then answers the same exact
// nearest-neighbour query at each voxel centre of the bake grid; the signed distance is |q - p| signed by
// the nearest point's normal and clipped like bake_sdf's output, so the field can be scored by eval_ao.py.
//
// DP-NNS follows the paper:
//   * BRIO order: random shuffle, levels of ratio 1/8 (each level ~7x all earlier ones, CGAL's default),
//     each level Hilbert-sorted;
//   * the successor DAG: when p is inserted into the Delaunay triangulation of the points inserted so far,
//     p is appended to the successor list of each of its new Delaunay neighbours (ranks = insertion order);
//   * level-consistent parallel build: small levels inserted sequentially, every later level inserted in
//     parallel with CGAL's vertex locks (priority blocking), a barrier between levels, a rank taken from a
//     shared counter while the insertion's locks are held, edges recorded per thread and assembled at the end;
//   * query: from the start site, move to the lowest-rank successor strictly closer to q; stop when none;
//   * grid initialisation: per cell, the highest-rank site common to the query paths of its 8 corners;
//   * queries in 3D Morton order (sort included in the query time).
//
//   pointnn_bench --boxes archive.json --lo -1 -1 -1 --span 26 18 8 --res 128 --clip 1.5 --points 300000
//                 --method dpnns|kdtree|bvh|brute --threads N [--grid G] [--out field.f32] [--validate K]
#define CGAL_LINKED_WITH_TBB 1
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/hilbert_sort.h>
#include <CGAL/Spatial_sort_traits_adapter_3.h>
#include <CGAL/property_map.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/task_group.h>
#include <nanoflann.hpp>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

struct P3 { float x, y, z; };
static inline float d2(const P3& a, const P3& b) { float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z; return dx * dx + dy * dy + dz * dz; }

// ------------------------------------------------------------------ scene and sampling
struct Box { double c[3], h[3]; };
static std::vector<Box> loadBoxes(const std::string& path) {
    std::ifstream in(path); if (!in) throw std::runtime_error("cannot read " + path);
    std::stringstream ss; ss << in.rdbuf(); std::string s = ss.str(); std::vector<double> v; const char* p = s.c_str();
    while (*p) { if ((*p >= '0' && *p <= '9') || *p == '-' || *p == '+' || *p == '.') { char* e; double x = std::strtod(p, &e); if (e != p) { v.push_back(x); p = e; continue; } } ++p; }
    std::vector<Box> b(v.size() / 6);
    for (size_t i = 0; i < b.size(); ++i) for (int k = 0; k < 3; ++k) { b[i].c[k] = v[6 * i + k]; b[i].h[k] = v[6 * i + 3 + k]; }
    return b;
}
// Area-uniform samples on all box faces; a sample is dropped when the surface there is covered by another
// box (a point just outside the face, along its normal, lies inside or on another box).
static void sampleBoxes(const std::vector<Box>& boxes, size_t target, uint32_t seed, std::vector<P3>& pts, std::vector<P3>& nrm) {
    struct Face { int box, axis, sgn; double area; };
    std::vector<Face> f; std::vector<double> cdf; double tot = 0;
    for (int i = 0; i < (int)boxes.size(); ++i) for (int a = 0; a < 3; ++a) for (int s = -1; s <= 1; s += 2) {
        int u = (a + 1) % 3, w = (a + 2) % 3; double ar = 4 * boxes[i].h[u] * boxes[i].h[w];
        if (ar <= 0) continue; f.push_back({i, a, s, ar}); tot += ar; cdf.push_back(tot);
    }
    std::mt19937_64 rng(seed); std::uniform_real_distribution<double> U(0, 1);
    pts.clear(); nrm.clear(); pts.reserve(target); nrm.reserve(target);
    for (size_t k = 0; k < target; ++k) {
        size_t fi = std::lower_bound(cdf.begin(), cdf.end(), U(rng) * tot) - cdf.begin(); if (fi >= f.size()) fi = f.size() - 1;
        const Face& F = f[fi]; const Box& B = boxes[F.box]; double p[3], n[3] = {0, 0, 0};
        int u = (F.axis + 1) % 3, w = (F.axis + 2) % 3;
        p[F.axis] = B.c[F.axis] + F.sgn * B.h[F.axis]; p[u] = B.c[u] + (2 * U(rng) - 1) * B.h[u]; p[w] = B.c[w] + (2 * U(rng) - 1) * B.h[w];
        n[F.axis] = F.sgn;
        double o[3]; for (int a = 0; a < 3; ++a) o[a] = p[a] + 1e-5 * n[a];
        bool covered = false;
        for (int j = 0; j < (int)boxes.size() && !covered; ++j) {
            if (j == F.box) continue; const Box& C = boxes[j]; bool in = true;
            for (int a = 0; a < 3 && in; ++a) in = std::fabs(o[a] - C.c[a]) <= C.h[a] + 1e-9;
            covered = in;
        }
        if (covered) continue;
        pts.push_back({float(p[0]), float(p[1]), float(p[2])}); nrm.push_back({float(n[0]), float(n[1]), float(n[2])});
    }
    // Drop exact duplicates (float rounding), so every method indexes the same distinct points.
    std::vector<uint32_t> o(pts.size()); for (size_t i = 0; i < o.size(); ++i) o[i] = uint32_t(i);
    auto key = [&](uint32_t i) { return std::make_tuple(pts[i].x, pts[i].y, pts[i].z); };
    std::sort(o.begin(), o.end(), [&](uint32_t a, uint32_t b) { return key(a) < key(b); });
    std::vector<P3> P2, N2; for (size_t k = 0; k < o.size(); ++k) if (!k || key(o[k]) != key(o[k - 1])) { P2.push_back(pts[o[k]]); N2.push_back(nrm[o[k]]); }
    // Restore a scene-independent order (seeded shuffle) so no method benefits from the sort.
    std::vector<uint32_t> s(P2.size()); for (size_t i = 0; i < s.size(); ++i) s[i] = uint32_t(i);
    std::shuffle(s.begin(), s.end(), rng);
    pts.resize(s.size()); nrm.resize(s.size()); for (size_t i = 0; i < s.size(); ++i) { pts[i] = P2[s[i]]; nrm[i] = N2[s[i]]; }
}

// ------------------------------------------------------------------ Morton order of queries
static inline uint64_t spread(uint32_t v) { uint64_t x = v & 0x1fffff; x = (x | x << 32) & 0x1f00000000ffffULL; x = (x | x << 16) & 0x1f0000ff0000ffULL; x = (x | x << 8) & 0x100f00f00f00f00fULL; x = (x | x << 4) & 0x10c30c30c30c30c3ULL; x = (x | x << 2) & 0x1249249249249249ULL; return x; }
// LSD radix sort of (key, index) by 30-bit Morton key (10 bits per axis), 3 passes of 10 bits.
static void mortonOrder(const std::vector<P3>& q, const P3& lo, const P3& span, std::vector<uint32_t>& order) {
    const size_t n = q.size(); std::vector<uint32_t> key(n), k2(n), o2(n); order.resize(n);
    tbb::parallel_for(size_t(0), n, [&](size_t i) {
        auto c = [&](float v, float l, float s) { int b = int((v - l) / s * 1024.f); return uint32_t(std::min(1023, std::max(0, b))); };
        key[i] = uint32_t(spread(c(q[i].x, lo.x, span.x)) | spread(c(q[i].y, lo.y, span.y)) << 1 | spread(c(q[i].z, lo.z, span.z)) << 2); order[i] = uint32_t(i);
    });
    for (int pass = 0; pass < 3; ++pass) {
        size_t cnt[1025] = {0}; int sh = pass * 10;
        for (size_t i = 0; i < n; ++i) ++cnt[((key[i] >> sh) & 1023) + 1];
        for (int b = 0; b < 1024; ++b) cnt[b + 1] += cnt[b];
        for (size_t i = 0; i < n; ++i) { size_t d = cnt[(key[i] >> sh) & 1023]++; k2[d] = key[i]; o2[d] = order[i]; }
        key.swap(k2); order.swap(o2);
    }
}

// ------------------------------------------------------------------ DP-NNS
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_with_info_3<uint32_t, K> Vb;
typedef CGAL::Delaunay_triangulation_cell_base_3<K> Cb;
typedef CGAL::Triangulation_data_structure_3<Vb, Cb, CGAL::Parallel_tag> Tds;
typedef CGAL::Delaunay_triangulation_3<K, Tds> DT;
typedef DT::Vertex_handle VH;

struct DPNNS {
    std::vector<P3> P;              // points by rank
    std::vector<uint32_t> src;      // P[rank] = input[src[rank]]
    std::vector<uint32_t> off, succ;// successor lists (CSR), each ascending by rank
    int G = 0; P3 glo{}, gcell{};   // grid
    std::vector<uint32_t> start;    // representative site per cell
    double tOrder = 0, tDT = 0, tAssemble = 0, tGrid = 0; size_t levels = 0, deferred = 0;

    void build(const std::vector<P3>& in, bool parallel, uint32_t seed) {
        auto t0 = clk::now(); const size_t n = in.size();
        std::vector<K::Point_3> pts(n); for (size_t i = 0; i < n; ++i) pts[i] = K::Point_3(in[i].x, in[i].y, in[i].z);
        std::vector<uint32_t> ord(n); for (size_t i = 0; i < n; ++i) ord[i] = uint32_t(i);
        std::mt19937 rng(seed); std::shuffle(ord.begin(), ord.end(), rng);
        // BRIO levels: [0, b1), [b1, b2), ... with b_{k} = b_{k+1} / 8, smallest level below 64 points.
        std::vector<size_t> bnd{n}; while (bnd.back() >= 64) bnd.push_back(size_t(double(bnd.back()) * 0.125)); bnd.push_back(0);
        std::reverse(bnd.begin(), bnd.end()); levels = bnd.size() - 1;
        typedef CGAL::Spatial_sort_traits_adapter_3<K, CGAL::Pointer_property_map<K::Point_3>::type> Tr;
        Tr tr(CGAL::make_property_map(pts));
        for (size_t l = 0; l + 1 < bnd.size(); ++l) {
            auto b = ord.begin() + bnd[l], e = ord.begin() + bnd[l + 1];
            if (parallel && e - b > 20000) CGAL::Hilbert_sort_3<Tr, CGAL::Hilbert_sort_median_policy, CGAL::Parallel_tag>(tr, 8)(b, e);
            else CGAL::Hilbert_sort_3<Tr, CGAL::Hilbert_sort_median_policy, CGAL::Sequential_tag>(tr, 8)(b, e);
        }
        auto t1 = clk::now(); tOrder = ms(t0, t1);

        CGAL::Bbox_3 bb = CGAL::bbox_3(pts.begin(), pts.end());
        DT::Lock_data_structure locks(bb, 50);
        DT dt(K(), parallel ? &locks : nullptr);
        std::atomic<uint32_t> rank{0}; src.assign(n, 0);
        struct Edge { uint32_t a, b; };
        tbb::enumerable_thread_specific<std::vector<Edge>> edges;
        tbb::enumerable_thread_specific<std::vector<VH>> nbuf;
        auto record = [&](VH v, uint32_t r, uint32_t input) {
            v->info() = r; src[r] = input;
            auto& nb = nbuf.local(); nb.clear(); dt.finite_adjacent_vertices(v, std::back_inserter(nb));
            auto& E = edges.local(); for (VH u : nb) E.push_back({u->info(), r});
        };
        VH hint;
        size_t i = 0, seqEnd = 0;
        // Sequential prefix: every level below 2000 points (and until the triangulation is 3D).
        for (size_t l = 0; l + 1 < bnd.size(); ++l) if (!parallel || bnd[l + 1] <= 2000) seqEnd = bnd[l + 1];
        for (; i < n && (i < seqEnd || dt.dimension() < 3); ++i) {
            size_t before = dt.number_of_vertices(); VH v = dt.insert(pts[ord[i]], hint);
            if (dt.number_of_vertices() == before) continue; // duplicate
            hint = v; record(v, rank++, ord[i]);
        }
        if (parallel) {
            tbb::enumerable_thread_specific<VH> tlsHint(hint);
            std::atomic<size_t> nDeferred{0};
            auto tryInsert = [&](uint32_t input, VH& h) -> bool {
                if (dt.try_lock_vertex(h) && dt.try_lock_point(pts[input])) {
                    bool ok; size_t before = dt.number_of_vertices();
                    VH v = dt.insert(pts[input], h, &ok);
                    if (ok) { if (dt.number_of_vertices() != before) record(v, rank++, input); h = v; }
                    dt.unlock_all_elements(); return ok;
                }
                dt.unlock_all_elements(); return false;
            };
            size_t lvlStart = i;
            for (size_t l = 0; l + 1 < bnd.size(); ++l) {
                size_t b = std::max(bnd[l], lvlStart), e = bnd[l + 1]; if (b >= e) continue;
                // Blocks of the level fetched by threads; a point that cannot get its locks is deferred and
                // retried after its block. Barrier between levels (end of parallel_for).
                tbb::parallel_for(tbb::blocked_range<size_t>(b, e, 256), [&](const tbb::blocked_range<size_t>& r) {
                    VH& h = tlsHint.local(); std::vector<uint32_t> later;
                    for (size_t k = r.begin(); k != r.end(); ++k) if (!tryInsert(ord[k], h)) later.push_back(ord[k]);
                    nDeferred += later.size();
                    while (!later.empty()) { std::vector<uint32_t> again; for (uint32_t x : later) if (!tryInsert(x, h)) again.push_back(x); later.swap(again); }
                });
            }
            deferred = nDeferred;
        }
        auto t2 = clk::now(); tDT = ms(t1, t2);
        // Assemble successor lists: bucket edges by source, sort each list by rank.
        const uint32_t m = rank; src.resize(m); P.resize(m);
        for (uint32_t r = 0; r < m; ++r) P[r] = in[src[r]];
        off.assign(size_t(m) + 1, 0);
        for (auto& E : edges) for (auto& e : E) ++off[e.a + 1];
        for (uint32_t r = 0; r < m; ++r) off[r + 1] += off[r];
        succ.resize(off[m]); std::vector<uint32_t> fill(off.begin(), off.end() - 1);
        for (auto& E : edges) for (auto& e : E) succ[fill[e.a]++] = e.b;
        tbb::parallel_for(uint32_t(0), m, [&](uint32_t r) { std::sort(succ.begin() + off[r], succ.begin() + off[r + 1]); });
        tAssemble = ms(t2, clk::now());
    }
    // The walk of Eq. 5; returns the nearest rank, and the visited transition sites when `path` is set.
    inline uint32_t walk(const P3& q, uint32_t c, size_t& evals, std::vector<uint32_t>* path = nullptr) const {
        float dc = d2(q, P[c]); ++evals; if (path) path->push_back(c);
        for (;;) {
            uint32_t next = UINT32_MAX;
            for (uint32_t j = off[c], e = off[c + 1]; j < e; ++j) {
                uint32_t s = succ[j]; float d = d2(q, P[s]); ++evals;
                if (d < dc) { next = s; dc = d; break; }
            }
            if (next == UINT32_MAX) return c;
            c = next; if (path) path->push_back(c);
        }
    }
    void buildGrid(int g, P3 lo, P3 span) {
        auto t0 = clk::now(); G = g; glo = lo; gcell = {span.x / g, span.y / g, span.z / g};
        start.assign(size_t(g) * g * g, 0);
        const int V = g + 1; std::vector<std::vector<uint32_t>> A(size_t(V) * V), B(size_t(V) * V);
        auto plane = [&](int k, std::vector<std::vector<uint32_t>>& out) {
            tbb::parallel_for(0, V * V, [&](int idx) {
                int i = idx % V, j = idx / V; P3 q{lo.x + i * gcell.x, lo.y + j * gcell.y, lo.z + k * gcell.z};
                size_t ev = 0; out[idx].clear(); walk(q, 0, ev, &out[idx]);
            });
        };
        plane(0, A);
        for (int k = 0; k < g; ++k) {
            plane(k + 1, B);
            tbb::parallel_for(0, g * g, [&](int idx) {
                int i = idx % g, j = idx / g;
                const std::vector<uint32_t>* c[8] = {&A[j * V + i], &A[j * V + i + 1], &A[(j + 1) * V + i], &A[(j + 1) * V + i + 1],
                                                     &B[j * V + i], &B[j * V + i + 1], &B[(j + 1) * V + i], &B[(j + 1) * V + i + 1]};
                uint32_t best = 0; // p1 (rank 0) is on every path
                for (auto it = c[0]->rbegin(); it != c[0]->rend(); ++it) {
                    bool all = true; for (int t = 1; t < 8 && all; ++t) all = std::binary_search(c[t]->begin(), c[t]->end(), *it);
                    if (all) { best = *it; break; }
                }
                start[(size_t(k) * g + j) * g + i] = best;
            });
            A.swap(B);
        }
        tGrid = ms(t0, clk::now());
    }
    inline uint32_t query(const P3& q, size_t& evals) const {
        uint32_t c = 0;
        if (G) {
            auto cl = [&](float v, float l, float s) { int b = int((v - l) / s); return std::min(G - 1, std::max(0, b)); };
            c = start[(size_t(cl(q.z, glo.z, gcell.z)) * G + cl(q.y, glo.y, gcell.y)) * G + cl(q.x, glo.x, gcell.x)];
        }
        return src[walk(q, c, evals)];
    }
};

// ------------------------------------------------------------------ BVH over points
struct PointBVH {
    struct Node { float lo[3], hi[3]; uint32_t right, begin, count; }; // count>0: leaf over idx[begin, begin+count)
    std::vector<Node> nodes; std::vector<uint32_t> idx; std::vector<P3> P; static constexpr uint32_t L = 4;
    static size_t nodesFor(size_t n) { return n <= L ? 1 : 1 + nodesFor(n / 2) + nodesFor(n - n / 2); }
    void build(const std::vector<P3>& in, bool parallel) {
        P = in; idx.resize(P.size()); for (size_t i = 0; i < idx.size(); ++i) idx[i] = uint32_t(i);
        nodes.resize(nodesFor(P.size())); tbb::task_group tg; rec(0, 0, P.size(), parallel ? &tg : nullptr); tg.wait();
        // Reorder points by leaf for locality.
        std::vector<P3> Q(P.size()); for (size_t i = 0; i < idx.size(); ++i) Q[i] = P[idx[i]]; P.swap(Q);
    }
    void rec(size_t ni, size_t b, size_t e, tbb::task_group* tg) {
        Node& nd = nodes[ni]; for (int k = 0; k < 3; ++k) { nd.lo[k] = INFINITY; nd.hi[k] = -INFINITY; }
        for (size_t i = b; i < e; ++i) { const float* p = &P[idx[i]].x; for (int k = 0; k < 3; ++k) { nd.lo[k] = std::min(nd.lo[k], p[k]); nd.hi[k] = std::max(nd.hi[k], p[k]); } }
        if (e - b <= L) { nd.count = uint32_t(e - b); nd.begin = uint32_t(b); nd.right = 0; return; }
        int ax = 0; float ext = -1; for (int k = 0; k < 3; ++k) if (nd.hi[k] - nd.lo[k] > ext) { ext = nd.hi[k] - nd.lo[k]; ax = k; }
        size_t n = e - b, m = b + n / 2;
        std::nth_element(idx.begin() + b, idx.begin() + m, idx.begin() + e, [&](uint32_t x, uint32_t y) { return (&P[x].x)[ax] < (&P[y].x)[ax]; });
        nd.count = 0; nd.right = uint32_t(ni + 1 + nodesFor(n / 2));
        size_t r = nd.right;
        if (tg && n > 50000) { tg->run([=] { rec(ni + 1, b, m, tg); }); rec(r, m, e, tg); }
        else { rec(ni + 1, b, m, tg); rec(r, m, e, tg); }
    }
    static inline float boxD2(const Node& n, const P3& q) {
        float dx = std::max(std::max(n.lo[0] - q.x, q.x - n.hi[0]), 0.f), dy = std::max(std::max(n.lo[1] - q.y, q.y - n.hi[1]), 0.f), dz = std::max(std::max(n.lo[2] - q.z, q.z - n.hi[2]), 0.f);
        return dx * dx + dy * dy + dz * dz;
    }
    inline uint32_t query(const P3& q, size_t& evals) const {
        uint32_t stack[64]; int sp = 0; uint32_t ni = 0; float best = INFINITY; uint32_t bi = 0;
        for (;;) {
            const Node& n = nodes[ni];
            if (n.count) {
                for (uint32_t i = n.begin; i < n.begin + n.count; ++i) { float d = d2(q, P[i]); ++evals; if (d < best) { best = d; bi = i; } }
            } else {
                uint32_t a = ni + 1, b = n.right; float da = boxD2(nodes[a], q), db = boxD2(nodes[b], q);
                if (db < da) { std::swap(a, b); std::swap(da, db); }
                if (db < best) stack[sp++] = b;       // farther child, re-checked on pop
                if (da < best) { ni = a; continue; }
            }
            for (;;) { if (!sp) return idx[bi]; ni = stack[--sp]; if (boxD2(nodes[ni], q) < best) break; }
        }
    }
};

// ------------------------------------------------------------------ kd-tree (nanoflann)
struct Cloud {
    const std::vector<P3>* p;
    inline size_t kdtree_get_point_count() const { return p->size(); }
    inline float kdtree_get_pt(size_t i, size_t d) const { return (&(*p)[i].x)[d]; }
    template <class BB> bool kdtree_get_bbox(BB&) const { return false; }
};
typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, Cloud>, Cloud, 3, uint32_t> KD;

// ------------------------------------------------------------------ main
int main(int argc, char** argv) {
    std::string boxesPath, method = "dpnns", out, statsPath; P3 lo{-1, -1, -1}, span{26, 18, 8}; int R = 128, threads = 8, G = 32;
    double clip = 1.5; size_t target = 300000, validate = 0; uint32_t seed = 1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--boxes") boxesPath = argv[++i];
        else if (a == "--lo") { lo.x = std::atof(argv[++i]); lo.y = std::atof(argv[++i]); lo.z = std::atof(argv[++i]); }
        else if (a == "--span") { span.x = std::atof(argv[++i]); span.y = std::atof(argv[++i]); span.z = std::atof(argv[++i]); }
        else if (a == "--res") R = std::atoi(argv[++i]);
        else if (a == "--clip") clip = std::atof(argv[++i]);
        else if (a == "--points") target = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--method") method = argv[++i];
        else if (a == "--threads") threads = std::atoi(argv[++i]);
        else if (a == "--grid") G = std::atoi(argv[++i]);
        else if (a == "--out") out = argv[++i];
        else if (a == "--stats") statsPath = argv[++i];
        else if (a == "--validate") validate = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--seed") seed = std::atoi(argv[++i]);
        else { std::cerr << "unknown option " << a << "\n"; return 2; }
    }
    tbb::global_control gc(tbb::global_control::max_allowed_parallelism, threads);
    std::vector<P3> pts, nrm; sampleBoxes(loadBoxes(boxesPath), target, seed, pts, nrm);
    // Queries: voxel centres, x fastest.
    const size_t nq = size_t(R) * R * R; std::vector<P3> q(nq);
    for (int z = 0; z < R; ++z) for (int y = 0; y < R; ++y) for (int x = 0; x < R; ++x)
        q[(size_t(z) * R + y) * R + x] = {lo.x + (x + .5f) * span.x / R, lo.y + (y + .5f) * span.y / R, lo.z + (z + .5f) * span.z / R};
    std::vector<uint32_t> nn(nq);
    std::atomic<size_t> evals{0};
    double tBuild = 0, tQuery = 0, tGrid = 0; std::string extra;
    auto runQueries = [&](auto&& one) {
        auto t0 = clk::now(); std::vector<uint32_t> order; mortonOrder(q, lo, span, order);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, nq, 4096), [&](const tbb::blocked_range<size_t>& r) {
            size_t ev = 0; for (size_t k = r.begin(); k != r.end(); ++k) { uint32_t i = order[k]; nn[i] = one(q[i], ev); } evals += ev;
        });
        tQuery = ms(t0, clk::now());
    };
    if (method == "dpnns") {
        DPNNS d; auto t0 = clk::now(); d.build(pts, threads > 1, seed); tBuild = ms(t0, clk::now());
        if (G) { d.buildGrid(G, lo, span); tGrid = d.tGrid; }
        runQueries([&](const P3& x, size_t& ev) { return d.query(x, ev); });
        char b[256]; std::snprintf(b, sizeof b, ",\"order_ms\":%.2f,\"dt_ms\":%.2f,\"assemble_ms\":%.2f,\"levels\":%zu,\"deferred\":%zu,\"edges\":%zu,\"grid\":%d", d.tOrder, d.tDT, d.tAssemble, d.levels, d.deferred, d.succ.size(), G);
        extra = b;
    } else if (method == "kdtree") {
        Cloud c{&pts}; auto t0 = clk::now();
        KD kd(3, c, nanoflann::KDTreeSingleIndexAdaptorParams(10, nanoflann::KDTreeSingleIndexAdaptorFlags::None, unsigned(threads)));
        tBuild = ms(t0, clk::now());
        runQueries([&](const P3& x, size_t& ev) { uint32_t i; float dd; nanoflann::KNNResultSet<float, uint32_t> rs(1); rs.init(&i, &dd); kd.findNeighbors(rs, &x.x); ++ev; return i; });
    } else if (method == "bvh") {
        PointBVH b; auto t0 = clk::now(); b.build(pts, threads > 1); tBuild = ms(t0, clk::now());
        runQueries([&](const P3& x, size_t& ev) { return b.query(x, ev); });
    } else if (method == "brute") {
        auto t0 = clk::now();
        tbb::parallel_for(size_t(0), nq, [&](size_t i) { float best = INFINITY; uint32_t bi = 0; for (uint32_t j = 0; j < pts.size(); ++j) { float d = d2(q[i], pts[j]); if (d < best) { best = d; bi = j; } } nn[i] = bi; });
        tQuery = ms(t0, clk::now());
    } else { std::cerr << "unknown method\n"; return 2; }

    // Exactness check against brute force on a deterministic subset of queries (distance ties allowed).
    size_t bad = 0;
    if (validate) {
        std::atomic<size_t> wrong{0}; size_t stride = std::max<size_t>(1, nq / validate);
        tbb::parallel_for(size_t(0), nq / stride, [&](size_t s) {
            size_t i = s * stride; float best = INFINITY; for (auto& p : pts) best = std::min(best, d2(q[i], p));
            if (d2(q[i], pts[nn[i]]) > best) ++wrong;
        });
        bad = wrong;
    }
    if (!out.empty()) {
        std::vector<float> f(nq);
        tbb::parallel_for(size_t(0), nq, [&](size_t i) {
            const P3& p = pts[nn[i]]; const P3& n = nrm[nn[i]];
            float d = std::sqrt(d2(q[i], p)); float s = (q[i].x - p.x) * n.x + (q[i].y - p.y) * n.y + (q[i].z - p.z) * n.z;
            f[i] = std::max(-float(clip), std::min(float(clip), s < 0 ? -d : d));
        });
        std::ofstream(out, std::ios::binary).write((const char*)f.data(), f.size() * 4);
    }
    char buf[512];
    std::snprintf(buf, sizeof buf, "{\"method\":\"%s\",\"points\":%zu,\"queries\":%zu,\"threads\":%d,\"build_ms\":%.2f,\"grid_ms\":%.2f,\"query_ms\":%.2f,\"us_per_query\":%.4f,\"evals_per_query\":%.1f,\"validated\":%zu,\"wrong\":%zu%s}",
                  method.c_str(), pts.size(), nq, threads, tBuild, tGrid, tQuery, tQuery * 1000 / nq, double(evals) / nq, validate ? nq / std::max<size_t>(1, nq / validate) : 0, bad, extra.c_str());
    std::cout << buf << std::endl;
    if (!statsPath.empty()) std::ofstream(statsPath) << buf << "\n";
    return bad ? 1 : 0;
}
