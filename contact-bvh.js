// Exact near-contact visibility for static triangle meshes. The cell grid selects
// a conservative local BVH; sphere tracing supplies farther visibility. Bounds
// come from mesh world transforms, never camera position or reference data.
// Rebuild after static vertex/index/instance transforms change. Dynamic geometry
// continues through the existing separate moving-object AO path.
export const CONTACT_GLSL=`
uniform sampler2D contactNodes;
uniform sampler2D contactTriangles;
uniform sampler3D contactGrid;
uniform vec3 contactGridMin;
uniform float contactGridSize;
uniform int contactNodeCount;
uniform float contactRange;
uniform float contactBias;
vec4 contactFetch(sampler2D tex,int i){int width=textureSize(tex,0).x;return texelFetch(tex,ivec2(i%width,i/width),0);}
bool contactBlocked(vec3 p,vec3 n,vec3 w){
 vec3 origin=p+n*contactBias,inv=1./(w+vec3(1e-20));
 int node=0,end=contactNodeCount;
 if(contactGridSize>0.){ivec3 cell=ivec3(floor((p-contactGridMin)/contactGridSize));ivec3 dims=textureSize(contactGrid,0);if(any(lessThan(cell,ivec3(0)))||any(greaterThanEqual(cell,dims)))return false;vec2 bounds=texelFetch(contactGrid,cell,0).xy;node=int(bounds.x);end=int(bounds.y);}
 for(int visit=0;visit<8192;visit++){
  if(node>=end)break;
  vec4 lo=contactFetch(contactNodes,node*2),hi=contactFetch(contactNodes,node*2+1);
  vec3 a=(lo.xyz-origin)*inv,b=(hi.xyz-origin)*inv;
  vec3 nearT=min(a,b),farT=max(a,b);
  float t0=max(0.,max(nearT.x,max(nearT.y,nearT.z)));
  float t1=min(contactRange,min(farT.x,min(farT.y,farT.z)));
  if(t1<t0){node=int(lo.w);continue;}
  int code=int(hi.w);
  if(code==0){node++;continue;}
  int first=code/8,count=code%8;
  for(int j=0;j<4;j++){
   if(j>=count)break;
   int tri=(first+j)*3;
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
  node=int(lo.w);
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
 // over sample index k: optimal 2D stratification at every frame count, zero mod-64 wrap.
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
// Prototype limits: at most 8192 nodes in one local tree, fewer than 2^21
// duplicated triangle records (float32 exact addressing), and device texture sizes.
export function buildContactBVH(objects,T,{gridSize=0,range=.25,maxTextureSize=4096,max3DTextureSize=256}={}){
 if(!Number.isFinite(range)||range<0||!Number.isFinite(gridSize)||gridSize<0)throw Error('Invalid contact BVH range/grid size');
 const triangles=[],v=new T.Vector3(),matrix=new T.Matrix4(),instance=new T.Matrix4();
 for(const object of objects){
  object.updateWorldMatrix(true,false);
  const g=object.geometry,pos=g.attributes.position,idx=g.index;
  for(let inst=0;inst<(object.isInstancedMesh?object.count:1);inst++){
   matrix.copy(object.matrixWorld);
   if(object.isInstancedMesh){object.getMatrixAt(inst,instance);matrix.multiply(instance);}
   for(let i=0;i<(idx?idx.count:pos.count);i+=3){
    const verts=[];
    for(let k=0;k<3;k++){v.fromBufferAttribute(pos,idx?idx.getX(i+k):i+k).applyMatrix4(matrix);verts.push(v.toArray());}
    const lo=[0,1,2].map(k=>Math.min(...verts.map(p=>p[k]))),hi=[0,1,2].map(k=>Math.max(...verts.map(p=>p[k])));
    triangles.push({verts,lo,hi,center:lo.map((x,k)=>(x+hi[k])/2)});
   }
  }
 }
 const nodes=[],ordered=[];
 function build(items){
  const at=nodes.length,node={lo:[Infinity,Infinity,Infinity],hi:[-Infinity,-Infinity,-Infinity],code:0,escape:0};nodes.push(node);
  for(const tri of items)for(let k=0;k<3;k++){node.lo[k]=Math.min(node.lo[k],tri.lo[k]);node.hi[k]=Math.max(node.hi[k],tri.hi[k]);}
  if(items.length<=4){node.code=ordered.length*8+items.length;ordered.push(...items);}
  else{let spans=node.hi.map((x,k)=>x-node.lo[k]),axis=spans.indexOf(Math.max(...spans));items.sort((a,b)=>a.center[axis]-b.center[axis]);let middle=items.length>>1;build(items.slice(0,middle));build(items.slice(middle));}
  node.escape=nodes.length;return at;
 }
 let gridData=new Float32Array(4),gridMin=[0,0,0],gridDims=[1,1,1],maxTreeNodes=0;
 if(gridSize>0&&triangles.length){
  const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
  for(const tri of triangles)for(let k=0;k<3;k++){lo[k]=Math.min(lo[k],tri.lo[k]);hi[k]=Math.max(hi[k],tri.hi[k]);}
  gridMin=lo.map(x=>Math.floor(x/gridSize)*gridSize);
  gridDims=hi.map((x,k)=>Math.floor((x-gridMin[k])/gridSize)+1);
  if(gridDims.some(n=>n>max3DTextureSize))throw Error(`Contact grid exceeds device 3D texture limit ${max3DTextureSize}`);
  gridData=new Float32Array(gridDims[0]*gridDims[1]*gridDims[2]*4);
  for(let z=0;z<gridDims[2];z++)for(let y=0;y<gridDims[1];y++)for(let x=0;x<gridDims[0];x++){
   const cell=[x,y,z],lower=cell.map((c,k)=>gridMin[k]+c*gridSize-range),upper=lower.map(v=>v+gridSize+2*range);
   const local=triangles.filter(t=>t.lo.every((v,k)=>v<=upper[k]&&t.hi[k]>=lower[k]));
   const first=nodes.length;if(local.length)build(local);maxTreeNodes=Math.max(maxTreeNodes,nodes.length-first);
   const offset=((z*gridDims[1]+y)*gridDims[0]+x)*4;gridData[offset]=first;gridData[offset+1]=nodes.length;
  }
 }else{if(triangles.length)build(triangles);maxTreeNodes=nodes.length;}
 if(maxTreeNodes>8192)throw Error('Contact BVH exceeds prototype traversal limit; increase the static loop bound.');
 function texture(values){const width=Math.min(1024,maxTextureSize,Math.max(1,values.length/4)),height=Math.ceil(values.length/4/width)||1;if(height>maxTextureSize)throw Error(`Contact BVH exceeds device texture limit ${maxTextureSize}`);const data=new Float32Array(width*height*4);data.set(values);const tex=new T.DataTexture(data,width,height,T.RGBAFormat,T.FloatType);tex.minFilter=tex.magFilter=T.NearestFilter;tex.needsUpdate=true;return tex;}
 if(ordered.length*8>=16777216)throw Error('BVH triangle addressing exceeds float32 exact integer range');
 const nodeValues=nodes.flatMap(n=>[...n.lo,n.escape,...n.hi,n.code]);
 const triValues=ordered.flatMap(t=>[...t.verts[0],0,...t.verts[1].map((x,k)=>x-t.verts[0][k]),0,...t.verts[2].map((x,k)=>x-t.verts[0][k]),0]);
 const nodeTexture=texture(nodeValues),triangleTexture=texture(triValues);
 const gridTexture=new T.Data3DTexture(gridData,...gridDims);gridTexture.format=T.RGBAFormat;gridTexture.type=T.FloatType;gridTexture.minFilter=gridTexture.magFilter=T.NearestFilter;gridTexture.needsUpdate=true;
 return {uniforms:{contactNodes:{value:nodeTexture},contactTriangles:{value:triangleTexture},contactNodeCount:{value:nodes.length},contactGrid:{value:gridTexture},contactGridMin:{value:new T.Vector3(...gridMin)},contactGridSize:{value:gridSize}},gridDims,maxTreeNodes,triangles:triangles.length,nodes:nodes.length,bytes:nodeTexture.image.data.byteLength+triangleTexture.image.data.byteLength+gridData.byteLength,dispose(){nodeTexture.dispose();triangleTexture.dispose();gridTexture.dispose();}};
}
