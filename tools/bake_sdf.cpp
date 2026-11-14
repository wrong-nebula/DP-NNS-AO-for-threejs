// bake_sdf: clipped signed distance volume for SDF ambient occlusion, from closed triangle meshes.
//
// Distances are exact (closest point on the nearest triangle, not on surface samples).
// Sign comes from the winding number along a z ray through every voxel column: each crossing
// counts +1 entering and -1 leaving (from the triangle's orientation), and a voxel is inside where
// the total is nonzero. Overlapping closed shells (posts through shelves) are handled without
// needing to know which triangles belong to which object. Meshes must be closed and consistently wound.
// Far space is skipped exactly: an 8^3 block whose centre is farther than clip + half-diagonal
// from every triangle is filled with +/-clip without per-voxel queries (distance is 1-Lipschitz).
//
// Input  --boxes scene.json   JSON array of [cx,cy,cz,hx,hy,hz] boxes
//        --obj   scene.obj    v / f lines (polygons are fanned)
// Output float32, x fastest, then y, then z; voxel i at lo + (i + 0.5) * span / res. Negative inside.
//
//   bake_sdf --boxes archive.json --lo -1 -1 -1 --span 26 18 8 --res 128 --clip 1.5 --out archive.f32
// Options: --threads N  --block B (0 disables culling)  --search bvh-neighbour|bvh|grid  --cell metres  --subdiv n (boxes)  --stats file.json
// Build:   c++ -O3 -std=c++17 -pthread bake_sdf.cpp -o bake_sdf    (no dependencies)
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using V3 = std::array<double,3>;
static V3 operator-(const V3& a,const V3& b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
static V3 operator+(const V3& a,const V3& b){return {a[0]+b[0],a[1]+b[1],a[2]+b[2]};}
static V3 operator*(const V3& a,double k){return {a[0]*k,a[1]*k,a[2]*k};}
static double dot3(const V3& a,const V3& b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
using Clk = std::chrono::steady_clock;
static double since(Clk::time_point t){return std::chrono::duration<double,std::milli>(Clk::now()-t).count();}

struct Tri { V3 a,b,c; int comp; };

// Squared distance from p to triangle abc (Ericson, Real-Time Collision Detection 5.1.5).
static double triDist2(const V3& p,const Tri& t){
    const V3 &a=t.a,&b=t.b,&c=t.c; V3 ab=b-a,ac=c-a,ap=p-a; double d1=dot3(ab,ap),d2=dot3(ac,ap);
    auto at=[&](const V3& q){V3 d=p-q;return dot3(d,d);};
    if(d1<=0&&d2<=0)return at(a);
    V3 bp=p-b; double d3=dot3(ab,bp),d4=dot3(ac,bp); if(d3>=0&&d4<=d3)return at(b);
    double vc=d1*d4-d3*d2; if(vc<=0&&d1>=0&&d3<=0){double v=d1/(d1-d3);return at(a+ab*v);}
    V3 cp=p-c; double d5=dot3(ab,cp),d6=dot3(ac,cp); if(d6>=0&&d5<=d6)return at(c);
    double vb=d5*d2-d1*d6; if(vb<=0&&d2>=0&&d6<=0){double w=d2/(d2-d6);return at(a+ac*w);}
    double va=d3*d6-d5*d4; if(va<=0&&(d4-d3)>=0&&(d5-d6)>=0){double w=(d4-d3)/((d4-d3)+(d5-d6));return at(b+(c-b)*w);}
    double den=1/(va+vb+vc),v=vb*den,w=vc*den; return at(a+ab*v+ac*w);
}

// ------------------------------------------------------------------ input
static std::vector<double> numbers(const std::string& s){
    std::vector<double> v; const char* p=s.c_str();
    while(*p){ if((*p>='0'&&*p<='9')||*p=='-'||*p=='+'||*p=='.'){char* e;double x=std::strtod(p,&e);if(e!=p){v.push_back(x);p=e;continue;}} ++p; }
    return v;
}
static std::vector<Tri> loadBoxes(const std::string& path,int subdiv=1){
    std::ifstream in(path); if(!in)throw std::runtime_error("cannot read "+path);
    std::stringstream ss; ss<<in.rdbuf(); auto v=numbers(ss.str());
    if(v.empty()||v.size()%6)throw std::runtime_error("expected a JSON array of [cx,cy,cz,hx,hy,hz]");
    std::vector<Tri> t;
    // 12 outward-wound triangles per box.
    static const int faces[6][4]={{0,2,3,1},{4,5,7,6},{0,1,5,4},{2,6,7,3},{0,4,6,2},{1,3,7,5}};
    for(size_t i=0;i<v.size();i+=6){
        V3 c{v[i],v[i+1],v[i+2]},h{v[i+3],v[i+4],v[i+5]};
        if(!(h[0]>0&&h[1]>0&&h[2]>0))throw std::runtime_error("box half-sizes must be positive");
        V3 p[8]; for(int k=0;k<8;++k)p[k]={c[0]+((k&1)?h[0]:-h[0]),c[1]+((k&2)?h[1]:-h[1]),c[2]+((k&4)?h[2]:-h[2])};
        // Each face as an n x n grid of quads (n = subdiv), to test dense meshes with a known exact answer.
        for(auto& f:faces){const V3 &q0=p[f[0]],&q1=p[f[1]],&q2=p[f[2]],&q3=p[f[3]];
            auto at=[&](double u,double v){return q0*((1-u)*(1-v))+q1*(u*(1-v))+q2*(u*v)+q3*((1-u)*v);};
            for(int a=0;a<subdiv;++a)for(int b=0;b<subdiv;++b){double u0=double(a)/subdiv,u1=double(a+1)/subdiv,v0=double(b)/subdiv,v1=double(b+1)/subdiv;
                t.push_back({at(u0,v0),at(u1,v0),at(u1,v1),int(i/6)});t.push_back({at(u0,v0),at(u1,v1),at(u0,v1),int(i/6)});}}
    }
    return t;
}
static std::vector<Tri> loadObj(const std::string& path){
    std::ifstream in(path); if(!in)throw std::runtime_error("cannot read "+path);
    std::vector<V3> v; std::vector<Tri> t; std::string line;
    while(std::getline(in,line)){
        if(line.rfind("v ",0)==0){auto n=numbers(line.substr(2));if(n.size()<3)throw std::runtime_error("bad vertex");v.push_back({n[0],n[1],n[2]});}
        else if(line.rfind("f ",0)==0){
            std::vector<int> f; std::istringstream ls(line.substr(2)); std::string tok;
            while(ls>>tok){int k=std::atoi(tok.c_str());if(k<0)k=int(v.size())+k+1;if(k<1||k>int(v.size()))throw std::runtime_error("bad face index");f.push_back(k-1);}
            for(size_t k=2;k<f.size();++k)t.push_back({v[f[0]],v[f[k-1]],v[f[k]],-1});
        }
    }
    return t;
}

// ------------------------------------------------------------------ triangle grid
struct TriGrid {
    V3 lo; double cell; std::array<int,3> dim;
    std::vector<uint32_t> start,items;
    TriGrid(const std::vector<Tri>& t,double cellSize){
        cell=cellSize; V3 hi; lo=t[0].a; hi=lo;
        for(auto& x:t)for(const V3* p:{&x.a,&x.b,&x.c})for(int k=0;k<3;++k){lo[k]=std::min(lo[k],(*p)[k]);hi[k]=std::max(hi[k],(*p)[k]);}
        for(int k=0;k<3;++k)dim[k]=std::max(1,int(std::ceil((hi[k]-lo[k])/cell))+1);
        start.assign(size_t(dim[0])*dim[1]*dim[2]+1,0);
        auto range=[&](const Tri& x,std::array<int,3>& a,std::array<int,3>& b){for(int k=0;k<3;++k){double mn=std::min({x.a[k],x.b[k],x.c[k]}),mx=std::max({x.a[k],x.b[k],x.c[k]});a[k]=cellOf(mn,k);b[k]=cellOf(mx,k);}};
        for(int pass=0;pass<2;++pass){
            std::vector<uint32_t> fill; if(pass){for(size_t i=1;i<start.size();++i)start[i]+=start[i-1];items.resize(start.back());fill.assign(start.begin(),start.end()-1);}
            for(uint32_t i=0;i<t.size();++i){std::array<int,3> a,b;range(t[i],a,b);
                for(int z=a[2];z<=b[2];++z)for(int y=a[1];y<=b[1];++y)for(int x=a[0];x<=b[0];++x){size_t c=(size_t(z)*dim[1]+y)*dim[0]+x;if(pass)items[fill[c]++]=i;else ++start[c+1];}}
        }
    }
    int cellOf(double v,int k) const {return std::clamp(int(std::floor((v-lo[k])/cell)),0,dim[k]-1);}
};
// Exact nearest triangle distance, searched shell by shell; returns +inf if none within `limit`.
// `bound` (if finite) is a known upper bound on the answer, e.g. from a seed triangle.
struct Searcher {
    const std::vector<Tri>& t; const TriGrid& g; std::vector<uint32_t> stamp; uint32_t q=0; uint64_t tested=0;
    Searcher(const std::vector<Tri>& tris,const TriGrid& grid):t(tris),g(grid),stamp(tris.size(),0){}
    double nearest(const V3& p,double limit,double bound=INFINITY){
        if(++q==0){std::fill(stamp.begin(),stamp.end(),0);q=1;}
        double best2=bound*bound; std::array<int,3> c; for(int k=0;k<3;++k)c[k]=g.cellOf(p[k],k);
        const int maxR=std::max({g.dim[0],g.dim[1],g.dim[2]});
        for(int r=0;r<=maxR;++r){
            // Lower bound on anything outside the visited (2r+1)^3 block.
            double outside=INFINITY;
            for(int k=0;k<3;++k){double a=p[k]-(g.lo[k]+(c[k]-r)*g.cell),b=g.lo[k]+(c[k]+r+1)*g.cell-p[k];
                if(c[k]-r>0)outside=std::min(outside,a); if(c[k]+r<g.dim[k]-1)outside=std::min(outside,b);}
            for(int z=c[2]-r;z<=c[2]+r;++z){if(z<0||z>=g.dim[2])continue;
             for(int y=c[1]-r;y<=c[1]+r;++y){if(y<0||y>=g.dim[1])continue;
              const bool edge=std::abs(z-c[2])==r||std::abs(y-c[1])==r;
              for(int x=c[0]-r;x<=c[0]+r;x+=(edge||r==0)?1:2*r){if(x<0||x>=g.dim[0])continue;
                const size_t cc=(size_t(z)*g.dim[1]+y)*g.dim[0]+x;
                for(uint32_t k=g.start[cc];k<g.start[cc+1];++k){uint32_t i=g.items[k];if(stamp[i]==q)continue;stamp[i]=q;++tested;best2=std::min(best2,triDist2(p,t[i]));}
              }}}
            const double best=std::sqrt(best2);
            if(best<=outside)return best<=limit?best:INFINITY;
            if(outside>limit)return INFINITY;      // nothing unvisited can be within the limit, and nothing visited was
        }
        const double best=std::sqrt(best2); return best<=limit?best:INFINITY;
    }
};

// ------------------------------------------------------------------ triangle BVH
// Binary BVH, median split on the longest centroid axis, leaves of up to 4 triangles.
// Closest-triangle queries visit the nearer child first and skip boxes beyond the best distance,
// so a good starting bound (e.g. the previous voxel's triangle) prunes most of the tree.
struct BVH {
    struct Node { V3 lo,hi; int left=-1,right=-1,start=0,count=0; };
    std::vector<Node> nodes; std::vector<uint32_t> order; const std::vector<Tri>& t;
    explicit BVH(const std::vector<Tri>& tris):t(tris){
        order.resize(t.size()); std::iota(order.begin(),order.end(),0);
        std::vector<V3> cen(t.size()); for(size_t i=0;i<t.size();++i)cen[i]=(t[i].a+t[i].b+t[i].c)*(1./3);
        nodes.reserve(2*t.size()/4+2);
        build(0,int(t.size()),cen);
    }
    int build(int s,int e,const std::vector<V3>& cen){
        Node n; n.lo={INFINITY,INFINITY,INFINITY}; n.hi={-INFINITY,-INFINITY,-INFINITY}; V3 clo=n.lo,chi=n.hi;
        for(int i=s;i<e;++i){const Tri& x=t[order[i]];for(const V3* p:{&x.a,&x.b,&x.c})for(int k=0;k<3;++k){n.lo[k]=std::min(n.lo[k],(*p)[k]);n.hi[k]=std::max(n.hi[k],(*p)[k]);}
            for(int k=0;k<3;++k){clo[k]=std::min(clo[k],cen[order[i]][k]);chi[k]=std::max(chi[k],cen[order[i]][k]);}}
        const int id=int(nodes.size()); nodes.push_back(n);
        if(e-s<=4){nodes[id].start=s;nodes[id].count=e-s;return id;}
        int ax=0; for(int k=1;k<3;++k)if(chi[k]-clo[k]>chi[ax]-clo[ax])ax=k;
        const int m=(s+e)/2;
        std::nth_element(order.begin()+s,order.begin()+m,order.begin()+e,[&](uint32_t a,uint32_t b){return cen[a][ax]<cen[b][ax];});
        const int l=build(s,m,cen),r=build(m,e,cen); nodes[id].left=l; nodes[id].right=r; return id;
    }
    static double boxDist2(const V3& p,const Node& n){double d=0;for(int k=0;k<3;++k){double v=std::max({n.lo[k]-p[k],0.,p[k]-n.hi[k]});d+=v*v;}return d;}
    // Returns the squared distance and sets `tri`; keeps `best2`/`tri` if nothing is closer.
    double closest(const V3& p,double best2,int& tri,uint64_t& tested) const {
        int stack[128]; int sp=0; stack[sp++]=0;
        while(sp){
            const Node& n=nodes[stack[--sp]];
            if(boxDist2(p,n)>=best2)continue;
            if(n.count){for(int i=0;i<n.count;++i){uint32_t k=order[n.start+i];++tested;double d=triDist2(p,t[k]);if(d<best2){best2=d;tri=int(k);}}continue;}
            const double dl=boxDist2(p,nodes[n.left]),dr=boxDist2(p,nodes[n.right]);
            if(dl<dr){if(dr<best2)stack[sp++]=n.right;if(dl<best2)stack[sp++]=n.left;}
            else{if(dl<best2)stack[sp++]=n.left;if(dr<best2)stack[sp++]=n.right;}
        }
        return best2;
    }
};

// ------------------------------------------------------------------ sign by winding number
// Inside = nonzero winding number along a +z ray through the voxel column centre.
static std::vector<uint8_t> insideMask(const std::vector<Tri>& t,const V3& lo,const V3& span,int R,int threads){
    std::vector<uint8_t> in(size_t(R)*R*R,0);
    // Bin triangles by their xy footprint in voxel-column units.
    const double sx=span[0]/R,sy=span[1]/R;
    std::vector<std::vector<uint32_t>> bins(size_t(R)*R);
    for(uint32_t i=0;i<t.size();++i){
        double x0=std::min({t[i].a[0],t[i].b[0],t[i].c[0]}),x1=std::max({t[i].a[0],t[i].b[0],t[i].c[0]});
        double y0=std::min({t[i].a[1],t[i].b[1],t[i].c[1]}),y1=std::max({t[i].a[1],t[i].b[1],t[i].c[1]});
        int i0=std::max(0,int(std::floor((x0-lo[0])/sx-.5))),i1=std::min(R-1,int(std::ceil((x1-lo[0])/sx-.5)));
        int j0=std::max(0,int(std::floor((y0-lo[1])/sy-.5))),j1=std::min(R-1,int(std::ceil((y1-lo[1])/sy-.5)));
        for(int j=j0;j<=j1;++j)for(int k=i0;k<=i1;++k)bins[size_t(j)*R+k].push_back(i);
    }
    std::atomic<int> next{0};
    auto work=[&]{
        std::vector<std::pair<double,int>> hits;
        for(int col;(col=next++)<R*R;){
            const int i=col%R,j=col/R;
            // Tiny irrational offset keeps the ray off shared edges and vertices of axis-aligned input.
            const double x=lo[0]+(i+.5)*sx+1.2345e-9*sx,y=lo[1]+(j+.5)*sy+2.7182e-9*sy;
            hits.clear();
            for(uint32_t k:bins[col]){
                const Tri& tr=t[k];
                double ax=tr.a[0]-x,ay=tr.a[1]-y,bx=tr.b[0]-x,by=tr.b[1]-y,cx=tr.c[0]-x,cy=tr.c[1]-y;
                double w0=bx*cy-by*cx,w1=cx*ay-cy*ax,w2=ax*by-ay*bx;
                if(!((w0>0&&w1>0&&w2>0)||(w0<0&&w1<0&&w2<0)))continue;
                // s > 0: triangle faces +z in projection, so a +z ray leaves through it (outward winding).
                const double s=w0+w1+w2; hits.push_back({(w0*tr.a[2]+w1*tr.b[2]+w2*tr.c[2])/s,s>0?-1:1});
            }
            if(hits.empty())continue;
            std::sort(hits.begin(),hits.end());
            int winding=0; size_t h=0;
            for(int z=0;z<R;++z){
                const double zc=lo[2]+(z+.5)*span[2]/R;
                while(h<hits.size()&&hits[h].first<zc)winding+=hits[h++].second;
                if(winding!=0)in[(size_t(z)*R+j)*R+i]=1;
            }
        }
    };
    std::vector<std::thread> pool; for(int k=0;k<threads;++k)pool.emplace_back(work); for(auto& p:pool)p.join();
    return in;
}

int main(int argc,char** argv){
    std::string boxes,obj,out,statsPath,search="bvh-neighbour"; int subdiv=1; double cellOverride=0; V3 lo{0,0,0},span{0,0,0}; int R=128,block=8,threads=int(std::max(1u,std::thread::hardware_concurrency())); double clip=1.5;
    for(int i=1;i<argc;++i){
        std::string a=argv[i]; auto need=[&](int n){if(i+n>=argc)throw std::runtime_error("missing value for "+a);};
        if(a=="--boxes"){need(1);boxes=argv[++i];}
        else if(a=="--obj"){need(1);obj=argv[++i];}
        else if(a=="--out"){need(1);out=argv[++i];}
        else if(a=="--stats"){need(1);statsPath=argv[++i];}
        else if(a=="--lo"){need(3);for(int k=0;k<3;++k)lo[k]=std::atof(argv[++i]);}
        else if(a=="--span"){need(3);for(int k=0;k<3;++k)span[k]=std::atof(argv[++i]);}
        else if(a=="--res"){need(1);R=std::atoi(argv[++i]);}
        else if(a=="--clip"){need(1);clip=std::atof(argv[++i]);}
        else if(a=="--threads"){need(1);threads=std::max(1,std::atoi(argv[++i]));}
        else if(a=="--block"){need(1);block=std::max(0,std::atoi(argv[++i]));}
        else if(a=="--search"){need(1);search=argv[++i];}
        else if(a=="--subdiv"){need(1);subdiv=std::max(1,std::atoi(argv[++i]));}
        else if(a=="--cell"){need(1);cellOverride=std::atof(argv[++i]);}
        else throw std::runtime_error("unknown option "+a);
    }
    if((boxes.empty()==obj.empty())||out.empty()||R<2||R>1024||!(span[0]>0&&span[1]>0&&span[2]>0)||!(clip>0))
        {std::cerr<<"Usage: bake_sdf (--boxes f.json | --obj f.obj) --lo x y z --span x y z --out field.f32 [--res 128] [--clip 1.5] [--threads N] [--block 8] [--search bvh-neighbour|bvh|grid] [--stats f.json]\n";return 2;}
    if(block&&R%block){std::cerr<<"--res must be a multiple of --block\n";return 2;}

    auto total=Clk::now(),t0=Clk::now();
    const auto tris=boxes.empty()?loadObj(obj):loadBoxes(boxes,subdiv);
    const double tLoad=since(t0);
    // Grid cell: about two mean edge lengths, but never finer than 1/256 of the extent or coarser than the clip.
    double edge=0; for(auto& x:tris)edge+=std::sqrt(dot3(x.b-x.a,x.b-x.a));
    edge/=tris.size();
    const double ext=std::max({span[0],span[1],span[2]});
    t0=Clk::now(); TriGrid grid(tris,cellOverride>0?cellOverride:std::clamp(2*edge,ext/256,clip/2)); const double tGrid=since(t0);
    t0=Clk::now(); BVH bvh(tris); const double tBvh=since(t0);
    // --search: bvh-neighbour (previous voxel's triangle as the starting bound, default) | bvh | grid
    const bool useGrid=search=="grid",neighbour=search=="bvh-neighbour";

    const auto seedBound=[](const V3&){return double(INFINITY);};
    if(!(useGrid||neighbour||search=="bvh"))throw std::runtime_error("unknown --search "+search);

    t0=Clk::now(); const auto inside=insideMask(tris,lo,span,R,threads); const double tSign=since(t0);

    std::vector<float> field(size_t(R)*R*R,float(clip));
    const V3 vox{span[0]/R,span[1]/R,span[2]/R};
    const int B=block?block:1,NB=R/B;
    const double halfDiag=.5*std::sqrt(dot3(vox,vox))*B;
    std::atomic<int> next{0}; std::atomic<uint64_t> queries{0},tested{0},culled{0};
    t0=Clk::now();
    auto work=[&]{
        Searcher s(tris,grid); uint64_t q=0,bt=0; int prevTri=-1;
        // Exact distance to the nearest triangle, or +inf if it is farther than `limit`.
        auto nearest=[&](const V3& p,double limit)->double{
            if(useGrid)return s.nearest(p,limit,std::min(seedBound(p),limit+1e-9));
            double start=limit+1e-9; int tri=-1;
            if(neighbour&&prevTri>=0){start=std::min(start,std::sqrt(triDist2(p,tris[prevTri])));tri=prevTri;}
            else if(!neighbour)start=std::min(start,seedBound(p));
            double d2=bvh.closest(p,start*start,tri,bt);
            if(tri>=0)prevTri=tri;
            const double d=std::sqrt(d2); return d<=limit?d:INFINITY;
        };
        for(int b;(b=next++)<NB*NB*NB;){
            const int bx=b%NB,by=(b/NB)%NB,bz=b/(NB*NB);
            if(block){
                const V3 c{lo[0]+(bx+.5)*B*vox[0],lo[1]+(by+.5)*B*vox[1],lo[2]+(bz+.5)*B*vox[2]};
                ++q;
                if(std::isinf(nearest(c,clip+halfDiag))){
                    // Every voxel is more than `clip` from the surface; no surface crossing inside the block.
                    for(int z=0;z<B;++z)for(int y=0;y<B;++y)for(int x=0;x<B;++x){size_t k=(size_t(bz*B+z)*R+by*B+y)*R+bx*B+x;field[k]=inside[k]?-float(clip):float(clip);}
                    ++culled; continue;
                }
            }
            for(int z=0;z<B;++z)for(int y=0;y<B;++y)for(int x=0;x<B;++x){
                const int i=bx*B+x,j=by*B+y,k=bz*B+z; const size_t idx=(size_t(k)*R+j)*R+i;
                const V3 p{lo[0]+(i+.5)*vox[0],lo[1]+(j+.5)*vox[1],lo[2]+(k+.5)*vox[2]};
                ++q; double d=nearest(p,clip);
                if(std::isinf(d))d=clip;
                field[idx]=inside[idx]?-float(d):float(std::min(d,clip));
            }
        }
        queries+=q; tested+=s.tested+bt;
    };
    std::vector<std::thread> pool; for(int k=0;k<threads;++k)pool.emplace_back(work); for(auto& p:pool)p.join();
    const double tField=since(t0);
    const double tTotal=since(total);

    std::ofstream(out,std::ios::binary).write((const char*)field.data(),field.size()*4);
    std::ostringstream js; js<<std::fixed<<std::setprecision(3)
      <<"{\"triangles\":"<<tris.size()<<",\"res\":"<<R<<",\"clip\":"<<clip<<",\"threads\":"<<threads<<",\"block\":"<<block<<",\"search\":\""<<search<<"\""
      <<",\"grid_cell\":"<<grid.cell<<",\"ms\":{\"load\":"<<tLoad<<",\"grid\":"<<tGrid<<",\"bvh\":"<<tBvh<<",\"sign\":"<<tSign<<",\"distance\":"<<tField<<",\"total\":"<<tTotal<<"}"
      <<",\"blocks\":"<<NB*NB*NB<<",\"blocks_culled\":"<<culled<<",\"distance_queries\":"<<queries<<",\"triangle_tests\":"<<tested<<"}";
    std::cout<<js.str()<<"\n";
    if(!statsPath.empty())std::ofstream(statsPath)<<js.str()<<"\n";
    return 0;
}
