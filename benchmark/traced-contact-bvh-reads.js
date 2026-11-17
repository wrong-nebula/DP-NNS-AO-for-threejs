// Instrumentation only: counts executed BVH and SDF texture fetches in the same
// candidate function, at the current geometry-buffer resolution and camera.
window.runContactBVHReadProbe=async({frames=[0,8,16,63]}={})=>{
 const d=aoDemo,T=d.T,R=d.renderer,original=d.aoPass.material.uniforms;
 d.render(64);
 const image=original.gPos.value.image,W=image.width,H=image.height;
 const uniforms={...original,frame:{value:0}};
 const contact=contactBVH.glsl.replace('return texelFetch(tex,','bvhReads+=1.;return texelFetch(tex,').replace('vec2 bounds=texelFetch(contactGrid,cell,0).xy;','bvhReads+=1.;vec2 bounds=texelFetch(contactGrid,cell,0).xy;');
 const shader=`precision highp float;precision highp sampler3D;
 uniform sampler2D gPos;uniform sampler2D gNormal;uniform sampler3D field;uniform float frame;
 float bvhReads=0.;float sdfReads=0.;
 float staticDistance(vec3 p){vec3 uv=(p+1.)/vec3(26.,18.,8.);if(any(lessThan(uv,vec3(0.)))||any(greaterThan(uv,vec3(1.))))return 1e4;sdfReads+=1.;return texture(field,uv).r;}
 float ign(vec2 q){return fract(52.9829189*fract(dot(q,vec2(.06711056,.00583715))));}
 ${contact}
 out vec4 probeOut;
 void main(){ivec2 q=ivec2(gl_FragCoord.xy);vec4 p=texelFetch(gPos,q,0),n=texelFetch(gNormal,q,0);if(p.w<=0.||n.w>.5){probeOut=vec4(0.);return;}float ao=candidateAO(p.xyz,normalize(n.xyz));probeOut=vec4(ao,bvhReads,sdfReads,1.);}`;
 const material=new T.ShaderMaterial({glslVersion:T.GLSL3,uniforms,vertexShader:'void main(){gl_Position=vec4(position.xy,0.,1.);}',fragmentShader:shader});
 const scene=new T.Scene(),geometry=new T.PlaneGeometry(2,2);scene.add(new T.Mesh(geometry,material));const camera=new T.Camera(),rt=new T.WebGLRenderTarget(W,H,{type:T.FloatType,format:T.RGBAFormat,depthBuffer:false}),buffer=new Float32Array(W*H*4),results={resolution:[W,H],frames:{}};
 for(const frame of frames){uniforms.frame.value=frame;R.setRenderTarget(rt);R.render(scene,camera);R.readRenderTargetPixels(rt,0,0,W,H,buffer);let n=0,bvh=0,sdf=0,maxBVH=0;for(let i=0;i<buffer.length;i+=4){if(!buffer[i+3])continue;n++;bvh+=buffer[i+1];sdf+=buffer[i+2];maxBVH=Math.max(maxBVH,buffer[i+1]);}results.frames[frame]={pixels:n,meanBVH:bvh/n,meanSDF:sdf/n,meanTotal:(bvh+sdf)/n,maxBVH};await new Promise(r=>setTimeout(r,0));}
 R.setRenderTarget(null);geometry.dispose();material.dispose();rt.dispose();await gpuValidation.save('bvh-read-counts.json',results);return results;
};
