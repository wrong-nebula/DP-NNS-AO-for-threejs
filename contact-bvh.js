// Exact near-contact visibility for static triangle meshes.
// Phase 3A: O(T) AABB-to-cell binning + binned SAH + cell-clipped quantized RGBA32UI
// single-fetch nodes (16 B/node) + ray-sign-ordered short-stack traversal.
// Bounds come from mesh world transforms, never camera position or reference data.
export const CONTACT_GLSL=`
uniform highp usampler2D contactNodes;
uniform highp usampler2D contactTriIdx;
uniform sampler2D contactTriangles;
uniform highp usampler3D contactGrid;
uniform vec3 contactGridMin;
uniform float contactGridSize;
uniform vec3 contactRootMin;
uniform vec3 contactRootSpan;
uniform int contactNodeCount;
uniform float contactRange;
uniform float contactBias;
vec4 contactFetch(sampler2D tex,int i){int width=textureSize(tex,0).x;return texelFetch(tex,ivec2(i%width,i/width),0);}
uvec4 contactFetchNode(int i){int width=textureSize(contactNodes,0).x;return texelFetch(contactNodes,ivec2(i%width,i/width),0);}
int contactFetchTriId(int i){int width=textureSize(contactTriIdx,0).x;return int(texelFetch(contactTriIdx,ivec2(i%width,i/width),0).r);}
uniform int contactMode;
uniform mat4 curViewProj;
uniform mat4 curView;
uniform sampler2D hizTex1;
uniform sampler2D hizTex2;
bool contactBlockedHiZ(vec3 p,vec3 n,vec3 w){
 vec3 p0=p+n*.008,p1=p+w*contactRange;
 vec4 c0=curViewProj*vec4(p0,1.),c1=curViewProj*vec4(p1,1.);
 if(c0.w<=0.||c1.w<=0.)return false;
 vec2 uv0=c0.xy/c0.w*.5+.5,uv1=c1.xy/c1.w*.5+.5;
 float z0=-(curView*vec4(p0,1.)).z,z1=-(curView*vec4(p1,1.)).z;
 vec2 res0=vec2(textureSize(gPos,0));
 vec2 res1=vec2(textureSize(hizTex1,0));
 float thickness=.045+.012*z0;
 // 8-step HZB-assisted contact march over [0, contactRange]
 for(int s=1;s<=8;s++){
  float t=float(s)/8.;
  // Perspective-correct screen UV and linear view depth along ray
  float invZ=mix(1./z0,1./z1,t);
  float rayZ=1./invZ;
  vec2 uv=mix(uv0/z0,uv1/z1,t)*rayZ;
  if(any(lessThan(uv,vec2(0.)))||any(greaterThan(uv,vec2(1.))))return false;
  // Coarse mip-1 min-depth rejection before full-res tap
  float minZ1=texelFetch(hizTex1,clamp(ivec2(uv*res1),ivec2(0),ivec2(res1)-1),0).r;
  if(minZ1<=0.||rayZ<minZ1+.003)continue;
  float surfZ=texelFetch(gPos,clamp(ivec2(uv*res0),ivec2(0),ivec2(res0)-1),0).w;
  if(surfZ>0.&&rayZ>surfZ+.004&&rayZ<surfZ+thickness)return true;
 }
 return false;
}
bool contactBlocked(vec3 p,vec3 n,vec3 w){
 if(contactMode==1)return contactBlockedHiZ(p,n,w);
 vec3 origin=p+n*contactBias,inv=1./(w+vec3(1e-20));
 int root=0,end=contactNodeCount;
 vec3 qMin=contactRootMin,qSpan=contactRootSpan;
 if(contactGridSize>0.){
  ivec3 cell=ivec3(floor((p-contactGridMin)/contactGridSize));
  ivec3 dims=textureSize(contactGrid,0);
  if(any(lessThan(cell,ivec3(0)))||any(greaterThanEqual(cell,dims)))return false;
  uvec2 bounds=texelFetch(contactGrid,cell,0).xy;
  root=int(bounds.x);end=int(bounds.y);
  if(root>=end)return false;
  qMin=contactGridMin+vec3(cell)*contactGridSize-vec3(contactRange+abs(contactBias));
  qSpan=vec3(contactGridSize+2.*(contactRange+abs(contactBias)));
 }else if(end<=0){return false;}
 vec3 qScale=qSpan*(1./65535.);
 int stack[24];
 int sp=0;
 int node=root;
 for(int visit=0;visit<2048;visit++){
  uvec4 nd=contactFetchNode(node);
  vec3 lo=qMin+vec3(float(nd.x&0xFFFFu),float(nd.x>>16u),float(nd.y&0xFFFFu))*qScale;
  vec3 hi=qMin+vec3(float(nd.y>>16u),float(nd.z&0xFFFFu),float(nd.z>>16u))*qScale;
  vec3 a=(lo-origin)*inv,b=(hi-origin)*inv;
  vec3 nearT=min(a,b),farT=max(a,b);
  float t0=max(0.,max(nearT.x,max(nearT.y,nearT.z)));
  float t1=min(contactRange,min(farT.x,min(farT.y,farT.z)));
  if(t1>=t0){
   uint meta=nd.w;
   if((meta&0x80000000u)!=0u){
    int first=int((meta&0x7FFFFFFFu)>>3u);
    int count=int(meta&7u);
    for(int j=0;j<4;j++){
     if(j>=count)break;
     int tri=contactFetchTriId(first+j)*3;
     vec3 v0=contactFetch(contactTriangles,tri).xyz;
     vec3 e1=contactFetch(contactTriangles,tri+1).xyz;
     vec3 e2=contactFetch(contactTriangles,tri+2).xyz;
     vec3 h=cross(w,e2);float det=dot(e1,h);
     if(abs(det)<1e-10)continue;
     vec3 s=origin-v0;float u=dot(s,h)/det;
     if(u<0.||u>1.)continue;
     vec3 q=cross(s,e1);float v=dot(w,q)/det;
     if(v<0.||u+v>1.)continue;
     float distance=dot(e2,q)/det;
     if(distance>0.&&distance<=contactRange)return true;
    }
   }else{
    int axis=int(meta>>29u);
    int right=int(meta&0x1FFFFFFFu);
    int left=node+1;
    bool neg=(axis==0?w.x:(axis==1?w.y:w.z))<0.;
    int firstChild=neg?right:left;
    int secondChild=neg?left:right;
    if(sp<24)stack[sp++]=secondChild;
    node=firstChild;
    continue;
   }
  }
  if(sp==0)break;
  node=stack[--sp];
 }
 return false;
}
float vdc2(uint bits){
 bits=(bits<<16u)|(bits>>16u);
 bits=((bits&0x55555555u)<<1u)|((bits&0xAAAAAAAAu)>>1u);
 bits=((bits&0x33333333u)<<2u)|((bits&0xCCCCCCCCu)>>2u);
 bits=((bits&0x0F0F0F0Fu)<<4u)|((bits&0xF0F0F0F0u)>>4u);
 bits=((bits&0x00FF00FFu)<<8u)|((bits&0xFF00FF00u)>>8u);
 return float(bits)*2.3283064365386963e-10;
}
float vdc3(uint n){
 float f=1.,r=0.;
 for(int j=0;j<6;j++){f/=3.;r+=f*float(n%3u);n/=3u;}
 return r;
}
float candidateAO(vec3 p,vec3 n,float nd){
 vec3 a=abs(n.x)<.9?vec3(1.,0.,0.):vec3(0.,1.,0.);
 vec3 t=normalize(cross(n,a)),b=cross(n,t);
 // Cranley-Patterson spatial blue-noise shift (IGN, constant per pixel) + exact 2D Halton(2,3)
 // over sample index: optimal 2D stratification at every frame count, zero mod-64 wrap.
 float su=ign(gl_FragCoord.xy);
 float sp=ign(gl_FragCoord.xy+vec2(37.,17.));
 float vis=0.;
 float fu=fract(su+vdc2(uint(frame)));
 uint base=uint(frame)*4u;
 for(int i=0;i<4;i++){
  if(float(i)>=nd)break;
  float u=clamp((float(i)+fu)/nd,1e-4,.9999);
  float phi=6.2831853*fract(sp+vdc3(base+uint(i)));
  vec3 w=t*(sqrt(u)*cos(phi))+b*(sqrt(u)*sin(phi))+n*sqrt(1.-u);
  if(contactBlocked(p,n,w))continue;
  float cosn=dot(w,n),s=contactRange,v=1.;
  for(int k=0;k<16;k++){
   float d=staticDistance(p+w*s),e=max(min(.025,.1*s*cosn),1e-6);
   v=min(v,clamp((d+e)/(4.*e),0.,1.));
   if(v<.01)break;
   s+=max(d,.072);if(s>1.12)break;
  }
  vis+=v;
 }
 return vis/nd;
}
float candidateAO(vec3 p,vec3 n){
 float nd=frame<8.?4.:(frame<16.?3.:2.);
 return candidateAO(p,n,nd);
}
`;

const SAH_BINS=8;
const bLoX=new Float32Array(SAH_BINS),bLoY=new Float32Array(SAH_BINS),bLoZ=new Float32Array(SAH_BINS);
const bHiX=new Float32Array(SAH_BINS),bHiY=new Float32Array(SAH_BINS),bHiZ=new Float32Array(SAH_BINS);
const bCnt=new Int32Array(SAH_BINS);
const rArea=new Float32Array(SAH_BINS),rCnt=new Int32Array(SAH_BINS);

export function buildContactBVH(objects,T,{gridSize=0,range=.25,maxTextureSize=4096,max3DTextureSize=256}={}){
 if(!Number.isFinite(range)||range<0||!Number.isFinite(gridSize)||gridSize<0)throw Error('Invalid contact BVH range/grid size');

 // Pass 1: count triangles so we allocate exact flat typed arrays once.
 let triCount=0;
 for(const object of objects){
  object.updateWorldMatrix(true,false);
  const g=object.geometry,idx=g.index,pos=g.attributes.position;
  const per=(idx?idx.count:pos.count)/3;
  triCount+=per*(object.isInstancedMesh?object.count:1);
 }

 const v0x=new Float32Array(triCount),v0y=new Float32Array(triCount),v0z=new Float32Array(triCount);
 const e1x=new Float32Array(triCount),e1y=new Float32Array(triCount),e1z=new Float32Array(triCount);
 const e2x=new Float32Array(triCount),e2y=new Float32Array(triCount),e2z=new Float32Array(triCount);
 const tLoX=new Float32Array(triCount),tLoY=new Float32Array(triCount),tLoZ=new Float32Array(triCount);
 const tHiX=new Float32Array(triCount),tHiY=new Float32Array(triCount),tHiZ=new Float32Array(triCount);
 const tCenX=new Float32Array(triCount),tCenY=new Float32Array(triCount),tCenZ=new Float32Array(triCount);

 let gLoX=Infinity,gLoY=Infinity,gLoZ=Infinity,gHiX=-Infinity,gHiY=-Infinity,gHiZ=-Infinity;
 const matrix=new T.Matrix4(),instance=new T.Matrix4();
 let ti=0;
 for(const object of objects){
  const g=object.geometry,pos=g.attributes.position,idx=g.index;
  const pArr=pos.array,pStride=pos.itemSize,iArr=idx?idx.array:null;
  const count=idx?idx.count:pos.count;
  const instCount=object.isInstancedMesh?object.count:1;
  for(let inst=0;inst<instCount;inst++){
   matrix.copy(object.matrixWorld);
   if(object.isInstancedMesh){object.getMatrixAt(inst,instance);matrix.multiply(instance);}
   const m=matrix.elements;
   const m0=m[0],m1=m[1],m2=m[2],m4=m[4],m5=m[5],m6=m[6],m8=m[8],m9=m[9],m10=m[10],m12=m[12],m13=m[13],m14=m[14];
   for(let i=0;i<count;i+=3){
    const a=(iArr?iArr[i]:i)*pStride,b=(iArr?iArr[i+1]:i+1)*pStride,c=(iArr?iArr[i+2]:i+2)*pStride;
    const ax=pArr[a],ay=pArr[a+1],az=pArr[a+2];
    const bx=pArr[b],by=pArr[b+1],bz=pArr[b+2];
    const cx=pArr[c],cy=pArr[c+1],cz=pArr[c+2];
    const x0=m0*ax+m4*ay+m8*az+m12,y0=m1*ax+m5*ay+m9*az+m13,z0=m2*ax+m6*ay+m10*az+m14;
    const x1=m0*bx+m4*by+m8*bz+m12,y1=m1*bx+m5*by+m9*bz+m13,z1=m2*bx+m6*by+m10*bz+m14;
    const x2=m0*cx+m4*cy+m8*cz+m12,y2=m1*cx+m5*cy+m9*cz+m13,z2=m2*cx+m6*cy+m10*cz+m14;
    v0x[ti]=x0;v0y[ti]=y0;v0z[ti]=z0;
    e1x[ti]=x1-x0;e1y[ti]=y1-y0;e1z[ti]=z1-z0;
    e2x[ti]=x2-x0;e2y[ti]=y2-y0;e2z[ti]=z2-z0;
    const lx=x0<x1?(x0<x2?x0:x2):(x1<x2?x1:x2);
    const ly=y0<y1?(y0<y2?y0:y2):(y1<y2?y1:y2);
    const lz=z0<z1?(z0<z2?z0:z2):(z1<z2?z1:z2);
    const hx=x0>x1?(x0>x2?x0:x2):(x1>x2?x1:x2);
    const hy=y0>y1?(y0>y2?y0:y2):(y1>y2?y1:y2);
    const hz=z0>z1?(z0>z2?z0:z2):(z1>z2?z1:z2);
    tLoX[ti]=lx;tLoY[ti]=ly;tLoZ[ti]=lz;
    tHiX[ti]=hx;tHiY[ti]=hy;tHiZ[ti]=hz;
    tCenX[ti]=(lx+hx)*0.5;tCenY[ti]=(ly+hy)*0.5;tCenZ[ti]=(lz+hz)*0.5;
    if(lx<gLoX)gLoX=lx;if(ly<gLoY)gLoY=ly;if(lz<gLoZ)gLoZ=lz;
    if(hx>gHiX)gHiX=hx;if(hy>gHiY)gHiY=hy;if(hz>gHiZ)gHiZ=hz;
    ti++;
   }
  }
 }

 // Growable flat buffers for quantized RGBA32UI nodes & ordered triangle indices.
 let nodeCap=Math.max(256,triCount*4),nodeLen=0;
 let nodeData=new Uint32Array(nodeCap*4);
 let ordCap=Math.max(256,triCount*4),ordLen=0;
 let ordIdx=new Int32Array(ordCap);

 function ensureNodes(extra){
  if(nodeLen+extra<=nodeCap)return;
  while(nodeLen+extra>nodeCap)nodeCap*=2;
  const next=new Uint32Array(nodeCap*4);next.set(nodeData.subarray(0,nodeLen*4));nodeData=next;
 }
 function ensureOrd(extra){
  if(ordLen+extra<=ordCap)return;
  while(ordLen+extra>ordCap)ordCap*=2;
  const next=new Int32Array(ordCap);next.set(ordIdx.subarray(0,ordLen));ordIdx=next;
 }

 // Working arrays for per-cell clipped triangle bounds/centroids + indices.
 let workCap=Math.max(256,triCount);
 let wIdx=new Int32Array(workCap);
 let wLoX=new Float32Array(workCap),wLoY=new Float32Array(workCap),wLoZ=new Float32Array(workCap);
 let wHiX=new Float32Array(workCap),wHiY=new Float32Array(workCap),wHiZ=new Float32Array(workCap);
 let wCX=new Float32Array(workCap),wCY=new Float32Array(workCap),wCZ=new Float32Array(workCap);

 let qMinX=0,qMinY=0,qMinZ=0,qInvX=1,qInvY=1,qInvZ=1;

 function packBox(at,lx,ly,lz,hx,hy,hz,meta){
  const qlx=Math.max(0,Math.min(65535,Math.floor((lx-qMinX)*qInvX)));
  const qly=Math.max(0,Math.min(65535,Math.floor((ly-qMinY)*qInvY)));
  const qlz=Math.max(0,Math.min(65535,Math.floor((lz-qMinZ)*qInvZ)));
  const qhx=Math.max(qlx,Math.min(65535,Math.ceil((hx-qMinX)*qInvX)));
  const qhy=Math.max(qly,Math.min(65535,Math.ceil((hy-qMinY)*qInvY)));
  const qhz=Math.max(qlz,Math.min(65535,Math.ceil((hz-qMinZ)*qInvZ)));
  const o=at*4;
  nodeData[o]=(qlx|(qly<<16))>>>0;
  nodeData[o+1]=(qlz|(qhx<<16))>>>0;
  nodeData[o+2]=(qhy|(qhz<<16))>>>0;
  nodeData[o+3]=meta>>>0;
 }

 function buildSubtree(s,e){
  ensureNodes(1);
  const at=nodeLen++;
  let lx=Infinity,ly=Infinity,lz=Infinity,hx=-Infinity,hy=-Infinity,hz=-Infinity;
  let clx=Infinity,cly=Infinity,clz=Infinity,chx=-Infinity,chy=-Infinity,chz=-Infinity;
  for(let i=s;i<e;i++){
   const k=wIdx[i];
   const a=wLoX[k],b=wLoY[k],c=wLoZ[k],d=wHiX[k],f=wHiY[k],g=wHiZ[k];
   if(a<lx)lx=a;if(b<ly)ly=b;if(c<lz)lz=c;
   if(d>hx)hx=d;if(f>hy)hy=f;if(g>hz)hz=g;
   const cx=wCX[k],cy=wCY[k],cz=wCZ[k];
   if(cx<clx)clx=cx;if(cy<cly)cly=cy;if(cz<clz)clz=cz;
   if(cx>chx)chx=cx;if(cy>chy)chy=cy;if(cz>chz)chz=cz;
  }
  const n=e-s;
  if(n<=4){
   ensureOrd(n);
   const first=ordLen;
   for(let i=s;i<e;i++)ordIdx[ordLen++]=wIdx[i];
   packBox(at,lx,ly,lz,hx,hy,hz,(0x80000000|((first<<3)|n))>>>0);
   return at;
  }

  // Binned SAH over all 3 axes.
  const extX=chx-clx,extY=chy-cly,extZ=chz-clz;
  let bestCost=Infinity,bestAx=-1,bestBin=0;
  for(let ax=0;ax<3;ax++){
   const cLo=ax===0?clx:(ax===1?cly:clz);
   const ext=ax===0?extX:(ax===1?extY:extZ);
   if(ext<=1e-7)continue;
   const sc=SAH_BINS/ext;
   const cArr=ax===0?wCX:(ax===1?wCY:wCZ);
   for(let b=0;b<SAH_BINS;b++){bCnt[b]=0;bLoX[b]=bLoY[b]=bLoZ[b]=Infinity;bHiX[b]=bHiY[b]=bHiZ[b]=-Infinity;}
   for(let i=s;i<e;i++){
    const k=wIdx[i];
    let bi=((cArr[k]-cLo)*sc)|0;
    if(bi>=SAH_BINS)bi=SAH_BINS-1;else if(bi<0)bi=0;
    bCnt[bi]++;
    if(wLoX[k]<bLoX[bi])bLoX[bi]=wLoX[k];if(wLoY[k]<bLoY[bi])bLoY[bi]=wLoY[k];if(wLoZ[k]<bLoZ[bi])bLoZ[bi]=wLoZ[k];
    if(wHiX[k]>bHiX[bi])bHiX[bi]=wHiX[k];if(wHiY[k]>bHiY[bi])bHiY[bi]=wHiY[k];if(wHiZ[k]>bHiZ[bi])bHiZ[bi]=wHiZ[k];
   }
   let rlx=Infinity,rly=Infinity,rlz=Infinity,rhx=-Infinity,rhy=-Infinity,rhz=-Infinity,rc=0;
   for(let b=SAH_BINS-1;b>0;b--){
    rc+=bCnt[b];
    if(bLoX[b]<rlx)rlx=bLoX[b];if(bLoY[b]<rly)rly=bLoY[b];if(bLoZ[b]<rlz)rlz=bLoZ[b];
    if(bHiX[b]>rhx)rhx=bHiX[b];if(bHiY[b]>rhy)rhy=bHiY[b];if(bHiZ[b]>rhz)rhz=bHiZ[b];
    rCnt[b]=rc;
    if(rc){const dx=rhx-rlx,dy=rhy-rly,dz=rhz-rlz;rArea[b]=dx*dy+dy*dz+dz*dx;}else rArea[b]=0;
   }
   let llx=Infinity,lly=Infinity,llz=Infinity,lhx=-Infinity,lhy=-Infinity,lhz=-Infinity,lc=0;
   for(let b=0;b<SAH_BINS-1;b++){
    lc+=bCnt[b];
    if(bLoX[b]<llx)llx=bLoX[b];if(bLoY[b]<lly)lly=bLoY[b];if(bLoZ[b]<llz)llz=bLoZ[b];
    if(bHiX[b]>lhx)lhx=bHiX[b];if(bHiY[b]>lhy)lhy=bHiY[b];if(bHiZ[b]>lhz)lhz=bHiZ[b];
    if(!lc||!rCnt[b+1])continue;
    const dx=lhx-llx,dy=lhy-lly,dz=lhz-llz;
    const cost=(dx*dy+dy*dz+dz*dx)*lc+rArea[b+1]*rCnt[b+1];
    if(cost<bestCost){bestCost=cost;bestAx=ax;bestBin=b+1;}
   }
  }

  let m=s;
  let splitAx=0;
  if(bestAx>=0){
   splitAx=bestAx;
   const cLo=bestAx===0?clx:(bestAx===1?cly:clz);
   const ext=bestAx===0?extX:(bestAx===1?extY:extZ);
   const sc=SAH_BINS/ext;
   const cArr=bestAx===0?wCX:(bestAx===1?wCY:wCZ);
   let i=s,j=e-1;
   while(i<=j){
    let bi=((cArr[wIdx[i]]-cLo)*sc)|0;if(bi>=SAH_BINS)bi=SAH_BINS-1;
    if(bi<bestBin){i++;}else{const tmp=wIdx[i];wIdx[i]=wIdx[j];wIdx[j]=tmp;j--;}
   }
   m=i;
  }
  if(m<=s||m>=e){
   const spX=hx-lx,spY=hy-ly,spZ=hz-lz;
   splitAx=spX>=spY&&spX>=spZ?0:(spY>=spZ?1:2);
   m=(s+e)>>1;
  }

  buildSubtree(s,m);
  const right=buildSubtree(m,e);
  packBox(at,lx,ly,lz,hx,hy,hz,(((splitAx&3)<<29)|(right&0x1FFFFFFF))>>>0);
  return at;
 }

 let gridData=new Uint32Array(2),gridMin=[0,0,0],gridDims=[1,1,1],maxTreeNodes=0;
 let rootMin=[0,0,0],rootSpan=[1,1,1];

 if(gridSize>0&&triCount>0){
  gridMin=[Math.floor(gLoX/gridSize)*gridSize,Math.floor(gLoY/gridSize)*gridSize,Math.floor(gLoZ/gridSize)*gridSize];
  gridDims=[Math.floor((gHiX-gridMin[0])/gridSize)+1,Math.floor((gHiY-gridMin[1])/gridSize)+1,Math.floor((gHiZ-gridMin[2])/gridSize)+1];
  if(gridDims[0]>max3DTextureSize||gridDims[1]>max3DTextureSize||gridDims[2]>max3DTextureSize)throw Error(`Contact grid exceeds device 3D texture limit ${max3DTextureSize}`);
  const gx=gridDims[0],gy=gridDims[1],gz=gridDims[2],nCells=gx*gy*gz;
  const invG=1/gridSize;

  // O(T) direct AABB-to-cell binning (2-pass CSR).
  const counts=new Int32Array(nCells);
  for(let i=0;i<triCount;i++){
   const x0=Math.max(0,Math.floor((tLoX[i]-range-gridMin[0])*invG));
   const x1=Math.min(gx-1,Math.floor((tHiX[i]+range-gridMin[0])*invG));
   const y0=Math.max(0,Math.floor((tLoY[i]-range-gridMin[1])*invG));
   const y1=Math.min(gy-1,Math.floor((tHiY[i]+range-gridMin[1])*invG));
   const z0=Math.max(0,Math.floor((tLoZ[i]-range-gridMin[2])*invG));
   const z1=Math.min(gz-1,Math.floor((tHiZ[i]+range-gridMin[2])*invG));
   for(let z=z0;z<=z1;z++)for(let y=y0;y<=y1;y++){
    let off=(z*gy+y)*gx+x0;
    for(let x=x0;x<=x1;x++,off++)counts[off]++;
   }
  }
  const offsets=new Int32Array(nCells+1);
  for(let c=0;c<nCells;c++)offsets[c+1]=offsets[c]+counts[c];
  const cellTris=new Int32Array(offsets[nCells]);
  const cursor=new Int32Array(offsets.subarray(0,nCells));
  for(let i=0;i<triCount;i++){
   const x0=Math.max(0,Math.floor((tLoX[i]-range-gridMin[0])*invG));
   const x1=Math.min(gx-1,Math.floor((tHiX[i]+range-gridMin[0])*invG));
   const y0=Math.max(0,Math.floor((tLoY[i]-range-gridMin[1])*invG));
   const y1=Math.min(gy-1,Math.floor((tHiY[i]+range-gridMin[1])*invG));
   const z0=Math.max(0,Math.floor((tLoZ[i]-range-gridMin[2])*invG));
   const z1=Math.min(gz-1,Math.floor((tHiZ[i]+range-gridMin[2])*invG));
   for(let z=z0;z<=z1;z++)for(let y=y0;y<=y1;y++){
    let off=(z*gy+y)*gx+x0;
    for(let x=x0;x<=x1;x++,off++)cellTris[cursor[off]++]=i;
   }
  }

  gridData=new Uint32Array(nCells*2);
  const cellSpan=gridSize+2*range;
  const invSpan=65535/cellSpan;
  for(let z=0;z<gz;z++)for(let y=0;y<gy;y++)for(let x=0;x<gx;x++){
   const cid=(z*gy+y)*gx+x;
   const cStart=offsets[cid],cEnd=offsets[cid+1],cLen=cEnd-cStart;
   const first=nodeLen;
   if(cLen>0){
    const clx=gridMin[0]+x*gridSize-range,cly=gridMin[1]+y*gridSize-range,clz=gridMin[2]+z*gridSize-range;
    const chx=clx+cellSpan,chy=cly+cellSpan,chz=clz+cellSpan;
    qMinX=clx;qMinY=cly;qMinZ=clz;qInvX=invSpan;qInvY=invSpan;qInvZ=invSpan;
    for(let i=0;i<cLen;i++){
     const k=cellTris[cStart+i];
     wIdx[i]=k;
     // Clip triangle AABB to expanded cell bounds (100% exact for rays starting in cell of length <= range).
     const lx=tLoX[k]>clx?tLoX[k]:clx,ly=tLoY[k]>cly?tLoY[k]:cly,lz=tLoZ[k]>clz?tLoZ[k]:clz;
     const hx=tHiX[k]<chx?tHiX[k]:chx,hy=tHiY[k]<chy?tHiY[k]:chy,hz=tHiZ[k]<chz?tHiZ[k]:chz;
     wLoX[k]=lx;wLoY[k]=ly;wLoZ[k]=lz;wHiX[k]=hx;wHiY[k]=hy;wHiZ[k]=hz;
     wCX[k]=(lx+hx)*0.5;wCY[k]=(ly+hy)*0.5;wCZ[k]=(lz+hz)*0.5;
    }
    buildSubtree(0,cLen);
    if(nodeLen-first>maxTreeNodes)maxTreeNodes=nodeLen-first;
   }
   gridData[cid*2]=first;gridData[cid*2+1]=nodeLen;
  }
 }else if(triCount>0){
  const pad=1e-4;
  rootMin=[gLoX-pad,gLoY-pad,gLoZ-pad];
  rootSpan=[Math.max(gHiX-gLoX+2*pad,1e-3),Math.max(gHiY-gLoY+2*pad,1e-3),Math.max(gHiZ-gLoZ+2*pad,1e-3)];
  qMinX=rootMin[0];qMinY=rootMin[1];qMinZ=rootMin[2];
  qInvX=65535/rootSpan[0];qInvY=65535/rootSpan[1];qInvZ=65535/rootSpan[2];
  for(let i=0;i<triCount;i++){
   wIdx[i]=i;wLoX[i]=tLoX[i];wLoY[i]=tLoY[i];wLoZ[i]=tLoZ[i];
   wHiX[i]=tHiX[i];wHiY[i]=tHiY[i];wHiZ[i]=tHiZ[i];
   wCX[i]=tCenX[i];wCY[i]=tCenY[i];wCZ[i]=tCenZ[i];
  }
  buildSubtree(0,triCount);
  maxTreeNodes=nodeLen;
 }

 if(ordLen*8>=0x7FFFFFFF)throw Error('BVH triangle addressing exceeds 28-bit leaf range');

 // Pack RGBA32UI node texture (16 B/node, 1 texel per node!)
 const nWidth=Math.min(1024,maxTextureSize,Math.max(1,nodeLen));
 const nHeight=Math.ceil(Math.max(1,nodeLen)/nWidth);
 if(nHeight>maxTextureSize)throw Error(`Contact BVH node texture exceeds device limit ${maxTextureSize}`);
 const nBuf=new Uint32Array(nWidth*nHeight*4);
 nBuf.set(nodeData.subarray(0,nodeLen*4));
 const nodeTexture=new T.DataTexture(nBuf,nWidth,nHeight,T.RGBAIntegerFormat,T.UnsignedIntType);
 nodeTexture.internalFormat='RGBA32UI';
 nodeTexture.minFilter=nodeTexture.magFilter=T.NearestFilter;
 nodeTexture.generateMipmaps=false;nodeTexture.needsUpdate=true;

 // Pack R32UI ordered triangle index indirection (4 B/ref) + deduplicated RGBA32F unique triangles (L1/L2 resident!)
 const idxTexels=Math.max(1,ordLen);
 const idxW=Math.min(1024,maxTextureSize,idxTexels);
 const idxH=Math.ceil(idxTexels/idxW);
 const idxBuf=new Uint32Array(idxW*idxH);
 for(let i=0;i<ordLen;i++)idxBuf[i]=ordIdx[i]>>>0;
 const triIdxTexture=new T.DataTexture(idxBuf,idxW,idxH,T.RedIntegerFormat,T.UnsignedIntType);
 triIdxTexture.internalFormat='R32UI';
 triIdxTexture.minFilter=triIdxTexture.magFilter=T.NearestFilter;
 triIdxTexture.generateMipmaps=false;triIdxTexture.needsUpdate=true;

 const tTexels=Math.max(1,triCount*3);
 const tWidth=Math.min(1024,maxTextureSize,tTexels);
 const tHeight=Math.ceil(tTexels/tWidth);
 if(tHeight>maxTextureSize)throw Error(`Contact BVH triangle texture exceeds device limit ${maxTextureSize}`);
 const tBuf=new Float32Array(tWidth*tHeight*4);
 for(let k=0;k<triCount;k++){
  const o=k*12;
  tBuf[o]=v0x[k];tBuf[o+1]=v0y[k];tBuf[o+2]=v0z[k];
  tBuf[o+4]=e1x[k];tBuf[o+5]=e1y[k];tBuf[o+6]=e1z[k];
  tBuf[o+8]=e2x[k];tBuf[o+9]=e2y[k];tBuf[o+10]=e2z[k];
 }
 const triangleTexture=new T.DataTexture(tBuf,tWidth,tHeight,T.RGBAFormat,T.FloatType);
 triangleTexture.minFilter=triangleTexture.magFilter=T.NearestFilter;
 triangleTexture.generateMipmaps=false;triangleTexture.needsUpdate=true;

 // Pack RG32UI 3D grid texture (8 B/cell instead of 16 B/cell!)
 const gridTexture=new T.Data3DTexture(gridData,...gridDims);
 gridTexture.format=T.RGIntegerFormat;gridTexture.type=T.UnsignedIntType;gridTexture.internalFormat='RG32UI';
 gridTexture.minFilter=gridTexture.magFilter=T.NearestFilter;gridTexture.unpackAlignment=1;gridTexture.needsUpdate=true;

 return {
  uniforms:{
   contactNodes:{value:nodeTexture},
   contactTriIdx:{value:triIdxTexture},
   contactTriangles:{value:triangleTexture},
   contactNodeCount:{value:nodeLen},
   contactGrid:{value:gridTexture},
   contactGridMin:{value:new T.Vector3(...gridMin)},
   contactGridSize:{value:gridSize},
   contactRootMin:{value:new T.Vector3(...rootMin)},
   contactRootSpan:{value:new T.Vector3(...rootSpan)},
  },
  gridDims,maxTreeNodes,triangles:triCount,nodes:nodeLen,
  bytes:nBuf.byteLength+idxBuf.byteLength+tBuf.byteLength+gridData.byteLength,
  dispose(){nodeTexture.dispose();triIdxTexture.dispose();triangleTexture.dispose();gridTexture.dispose();}
 };
}
