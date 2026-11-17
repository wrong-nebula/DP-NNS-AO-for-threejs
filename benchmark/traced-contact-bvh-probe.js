// Standalone geometry stress test. Reference is independent CPU ray/box visibility;
// it is never consulted by the renderer or BVH builder.
window.runContactBVHProbe=async({gaps=[.02,.04,.08,.12,.20],referenceRays=16384}={})=>{
 const d=aoDemo,T=d.T,R=d.renderer,W=129,H=16;
 const trace=d.aoPass.material.fragmentShader.split('float ign(')[1].split('// The previous frame')[0];
 const shader=`precision highp float;precision highp sampler3D;
 uniform float frame;uniform float validationStart;uniform float validationFront;uniform float validationClamp;uniform float gap;
 float staticDistance(vec3 p){vec3 q=abs(p-vec3(0.,0.,gap+.01))-vec3(.25,.25,.01);return min(p.z,length(max(q,0.))+min(max(q.x,max(q.y,q.z)),0.));}
 float ign(${trace}
 out vec4 probeOut;
 void main(){vec3 p=vec3((gl_FragCoord.x-.5)/128.*.8-.4,0.,0.);probeOut=vec4(tracedAO(p,vec3(0.,0.,1.)),0.,0.,1.);}`;
 const u={frame:{value:0},validationStart:{value:.02},validationFront:{value:0},validationClamp:{value:1e-6},gap:{value:.02},contactRange:{value:d.aoPass.material.uniforms.contactRange.value},contactBias:{value:d.aoPass.material.uniforms.contactBias.value}};
 const material=new T.ShaderMaterial({glslVersion:T.GLSL3,defines:{DIRS:2},uniforms:u,vertexShader:'void main(){gl_Position=vec4(position.xy,0.,1.);}',fragmentShader:shader});
 const scene=new T.Scene();scene.add(new T.Mesh(new T.PlaneGeometry(2,2),material));const camera=new T.Camera(),rt=new T.WebGLRenderTarget(W,H,{type:T.FloatType,format:T.RGBAFormat,depthBuffer:false});
 const xs=Array.from({length:W},(_,i)=>i/128*.8-.4);
 const output={gaps,xs,referenceRays,contactRange:u.contactRange.value,bias:u.contactBias.value,profiles:{},reference:[]};
 const dirs=Array.from({length:referenceRays},(_,i)=>{const z=Math.sqrt(1-(i+.5)/referenceRays),r=Math.sqrt((i+.5)/referenceRays),phi=i*2.399963229728653;return [r*Math.cos(phi),r*Math.sin(phi),z];});
 function reference(x,gap){let hits=0;for(const [wx,wy,wz] of dirs){const a=(-.25-x)/wx,b=(.25-x)/wx,c=-.25/wy,e=.25/wy;const enter=Math.max(0,Math.min(a,b),Math.min(c,e),gap/wz),exit=Math.min(1.12,Math.max(a,b),Math.max(c,e),(gap+.02)/wz);if(exit>=enter)hits++;}return 1-hits/referenceRays;}
 for(let gi=0;gi<gaps.length;gi++){
  const gap=gaps[gi];u.gap.value=gap;
  const plate=new T.Mesh(new T.BoxGeometry(.5,.5,.02));plate.position.z=gap+.01;
  const floor=new T.Mesh(new T.BoxGeometry(4,4,.1));floor.position.z=-.05;
  const tree=contactBVH.build([floor,plate],T);Object.assign(u,tree.uniforms);material.needsUpdate=true;
  output.reference.push(xs.map(x=>reference(x,gap)));
  for(const [name,params] of Object.entries({seed:[.02,0],champion:[.12,1],next:[.12,2]})){
   u.validationStart.value=params[0];u.validationFront.value=params[1];
   const sums=new Float64Array(W),buf=new Float32Array(W*H*4),checkpoints={};
   for(let f=0;f<64;f++){
    u.frame.value=f;R.setRenderTarget(rt);R.render(scene,camera);R.readRenderTargetPixels(rt,0,0,W,H,buf);
    for(let y=0;y<H;y++)for(let x=0;x<W;x++)sums[x]+=buf[(y*W+x)*4]/H;
    if([1,4,16,64].includes(f+1))checkpoints[f+1]=Array.from(sums,v=>v/(f+1));
    if(f%8===0)await new Promise(r=>setTimeout(r,0));
   }
   (output.profiles[name]??=[]).push(checkpoints);
  }
  tree.dispose();plate.geometry.dispose();floor.geometry.dispose();
 }
 R.setRenderTarget(null);rt.dispose();material.dispose();scene.children[0].geometry.dispose();
 output.summary=Object.fromEntries(Object.entries(output.profiles).map(([name,sets])=>[name,sets.map((checkpoints,i)=>({gap:gaps[i],center64:checkpoints[64][64],referenceCenter:output.reference[i][64],mae64:checkpoints[64].reduce((s,v,j)=>s+Math.abs(v-output.reference[i][j]),0)/W,tooLight64:checkpoints[64].reduce((s,v,j)=>s+Math.max(0,v-output.reference[i][j]),0)/W}))]));
 await gpuValidation.save('geometry-thin-plate-probe.json',output);return output.summary;
};
