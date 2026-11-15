// Mover benchmark harness for index.html (and benchmark/index.orig.html).
// Paste or inject into the page after it has loaded, then:
//   await moverBench.run({tag:'orig'})        // times + stores AO masks and atlas slabs in IndexedDB
//   await moverBench.run({tag:'cand', compareTo:'orig'})
// GPU times follow benchmark/final.html: 12 draws per TIME_ELAPSED query, median over rounds.
// Poses are frozen (updateMovers(t) at fixed t), the camera is the benchmark camera, and the
// render loop is paused, so both pages draw exactly the same frames.
(()=>{
const POSES=[0,3.7,7.4,11.1,14.8,18.5,22.2,25.9];
const MODES=[14,4,34];
const db=()=>new Promise((ok,err)=>{const r=indexedDB.open('moverBench',1);r.onupgradeneeded=()=>r.result.createObjectStore('kv');r.onsuccess=()=>ok(r.result);r.onerror=()=>err(r.error);});
async function put(k,v){const d=await db();return new Promise((ok,err)=>{const t=d.transaction('kv','readwrite');t.objectStore('kv').put(v,k);t.oncomplete=ok;t.onerror=()=>err(t.error);});}
async function get(k){const d=await db();return new Promise((ok,err)=>{const r=d.transaction('kv').objectStore('kv').get(k);r.onsuccess=()=>ok(r.result);r.onerror=()=>err(r.error);});}
const median=a=>{const s=[...a].sort((x,y)=>x-y);return s[s.length>>1];};
const q95=a=>{const s=[...a].sort((x,y)=>x-y);return s[Math.min(s.length-1,Math.floor(.95*(s.length-1)))];};
function setup(){
  const d=window.aoDemo; if(!d)throw Error('aoDemo missing');
  d.renderer.setAnimationLoop(null);
  d.camera.position.set(18,-4,7.3); d.controls.target.set(10.8,8.8,1.4); d.controls.update();
  return d;
}
async function gpuTime(d,gl,timer,mode){
  const q=gl.createQuery(); gl.beginQuery(timer.TIME_ELAPSED_EXT,q);
  for(let i=0;i<12;i++)d.render(mode);
  gl.endQuery(timer.TIME_ELAPSED_EXT); gl.flush();
  try{for(let i=0;i<2000;i++){if(gl.getQueryParameter(q,gl.QUERY_RESULT_AVAILABLE)){
      if(gl.getParameter(timer.GPU_DISJOINT_EXT))return NaN;
      return gl.getQueryParameter(q,gl.QUERY_RESULT)/12e6;}
    await new Promise(r=>setTimeout(r,2));}
    throw Error('GPU timer timed out');}finally{gl.deleteQuery(q);}
}
function readMask(d,gl,mode){
  d.setMask(true); d.render(mode); d.setMask(false);
  const px=new Uint8Array(1280*720*4); gl.readPixels(0,0,1280,720,gl.RGBA,gl.UNSIGNED_BYTE,px);
  const r=new Uint8Array(1280*720); for(let i=0;i<r.length;i++)r[i]=px[4*i]; return r;
}
// Read every layer of the given atlas slabs as float (R channel).
function readAtlas(d,gl,slabs){
  const tex=d.renderer.properties.get(d.atlas).__webglTexture; if(!tex)return null;
  const R=d.meshRes, fb=gl.createFramebuffer(), out={}; const prev=gl.getParameter(gl.FRAMEBUFFER_BINDING);
  gl.bindFramebuffer(gl.FRAMEBUFFER,fb); const buf=new Float32Array(R*R*4);
  for(const s of slabs){const a=new Float32Array(R*R*R);
    for(let z=0;z<R;z++){gl.framebufferTextureLayer(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,tex,0,s.layer*R+z);
      gl.readPixels(0,0,R,R,gl.RGBA,gl.FLOAT,buf); for(let i=0;i<R*R;i++)a[z*R*R+i]=buf[4*i];}
    out[s.name]=a;}
  gl.bindFramebuffer(gl.FRAMEBUFFER,prev); gl.deleteFramebuffer(fb); return out;
}
async function run({tag,compareTo,rounds=25,store=true}={}){
  const d=setup(), gl=d.renderer.getContext(), timer=gl.getExtension('EXT_disjoint_timer_query_webgl2');
  if(!timer)throw Error('EXT_disjoint_timer_query_webgl2 unavailable');
  const res={tag,bakeMs:d.bakeMs,times:{},masks:{}};
  const cases=[];
  for(const mode of MODES){cases.push({k:`m${mode}_movers`,mode,dyn:true});cases.push({k:`m${mode}_static`,mode,dyn:false});}
  cases.push({k:'off',mode:0,dyn:true});
  for(const c of cases)res.times[c.k]=[];
  // warm-up
  for(const c of cases){d.setDynamicAO(c.dyn);d.updateMovers(POSES[0]);for(let i=0;i<12;i++)d.render(c.mode);} gl.finish();
  for(let r=0;r<rounds;r++){const t=POSES[r%POSES.length];
    for(let j=0;j<cases.length;j++){const c=cases[(r+j)%cases.length];d.setDynamicAO(c.dyn);d.updateMovers(t);res.times[c.k].push(await gpuTime(d,gl,timer,c.mode));}}
  if(rounds>0)res.summary=Object.fromEntries(Object.entries(res.times).map(([k,v])=>[k,{p50:+median(v).toFixed(4),p95:+q95(v).toFixed(4)}]));
  // masks at every pose, movers on, both modes
  d.setDynamicAO(true);
  const masks={};
  for(const mode of MODES)for(const t of POSES){d.updateMovers(t);masks[`${mode}@${t}`]=readMask(d,gl,mode);}
  d.setDynamicAO(false);
  for(const mode of MODES){d.updateMovers(POSES[0]);masks[`${mode}@static`]=readMask(d,gl,mode);}
  d.setDynamicAO(true);
  const meshSlabs=d.movers.map((m,i)=>({name:m.name,layer:m.meshParam?m.meshParam.x:-1,type:m.type})).filter(s=>s.type===6&&s.layer>=0);
  const atlas=readAtlas(d,gl,meshSlabs);
  if(store){await put(tag+':masks',masks);await put(tag+':atlas',atlas);}
  if(compareTo){
    const ref=await get(compareTo+':masks'), refAtlas=await get(compareTo+':atlas'); const cmp={};
    let diffPx=0,maxDiff=0,total=0;
    for(const k in masks){const a=masks[k],b=ref[k];let n=0,m=0;for(let i=0;i<a.length;i++){const e=Math.abs(a[i]-b[i]);if(e){n++;if(e>m)m=e;}}cmp[k]={diffPx:n,max:m};diffPx+=n;maxDiff=Math.max(maxDiff,m);total+=a.length;}
    const at={};for(const k in atlas){const a=atlas[k],b=refAtlas[k];let n=0,m=0;for(let i=0;i<a.length;i++){const e=Math.abs(a[i]-b[i]);if(e){n++;m=Math.max(m,e);}}at[k]={diffVoxels:n,maxAbs:m};}
    res.compare={against:compareTo,diffPixels:diffPx,totalPixels:total,max255:maxDiff,perPose:cmp,atlas:at};
  }
  delete res.times; res.timesRaw=undefined;
  d.setDynamicAO(true);
  return res;
}
// Interleaved A/B: load each URL in its own same-origin iframe, inject this harness, and
// alternate short timing runs so clock and thermal drift hit both pages equally.
//   await moverBench.ab(['/benchmark/index.orig.html','/index.html'],{blocks:8,rounds:5})
async function ab(urls,{blocks=8,rounds=5}={}){
  const frames=[];
  for(const u of urls){const f=document.createElement('iframe');f.style.cssText='position:fixed;left:0;top:0;width:1300px;height:900px;';
    f.src=u+(u.includes('?')?'&':'?')+'ab='+Math.random();document.body.append(f);
    for(let j=0;j<400&&!(f.contentWindow&&f.contentWindow.aoDemo);j++)await new Promise(r=>setTimeout(r,50));
    if(!f.contentWindow.aoDemo)throw Error('no aoDemo in '+u);
    await new Promise((ok,err)=>{const s=f.contentDocument.createElement('script');s.src='/benchmark/mover-bench.js?x='+Math.random();s.onload=ok;s.onerror=err;f.contentDocument.head.append(s);});
    frames.push(f);}
  const acc=urls.map(()=>({}));
  for(let b=0;b<blocks;b++)for(let k=0;k<urls.length;k++){const idx=(b+k)%urls.length;
    const w=frames[idx].contentWindow, r=await w.moverBench.runTimes({rounds});
    for(const [key,v] of Object.entries(r))(acc[idx][key]=acc[idx][key]||[]).push(...v);}
  frames.forEach(f=>f.remove());
  return Object.fromEntries(urls.map((u,i)=>[u,Object.fromEntries(Object.entries(acc[i]).map(([k,v])=>[k,{p50:+median(v).toFixed(4),p95:+q95(v).toFixed(4),n:v.length}]))]));
}
async function runTimes({rounds=5}={}){
  const d=setup(), gl=d.renderer.getContext(), timer=gl.getExtension('EXT_disjoint_timer_query_webgl2');
  const cases=[{k:'m14_movers',mode:14,dyn:true},{k:'m14_static',mode:14,dyn:false},{k:'m4_movers',mode:4,dyn:true},{k:'off',mode:0,dyn:true}];
  const out=Object.fromEntries(cases.map(c=>[c.k,[]]));
  for(const c of cases){d.setDynamicAO(c.dyn);d.updateMovers(POSES[0]);for(let i=0;i<12;i++)d.render(c.mode);} gl.finish();
  for(let r=0;r<rounds;r++){const t=POSES[(r*3)%POSES.length];
    for(let j=0;j<cases.length;j++){const c=cases[(r+j)%cases.length];d.setDynamicAO(c.dyn);d.updateMovers(t);const v=await gpuTime(d,gl,timer,c.mode);if(Number.isFinite(v))out[c.k].push(v);}}
  d.setDynamicAO(true);
  return out;
}
// MAE of stored GPU masks (tag from run()) against the ray-traced mover reference
// (export_reference_movers.py), at the reference pixels: x = 2+4i, y = 2+4j (top row first).
async function refMAE(tag,poses=['0.0','11.1']){
  const masks=await get(tag+':masks'), out={};
  const stat=new Float32Array(await (await fetch('/benchmark/ref_static.f32')).arrayBuffer());
  for(const ps of poses){
    const ref=new Float32Array(await (await fetch(`/benchmark/ref_movers_${ps}.f32`)).arrayBuffer());
    const flg=new Uint8Array(await (await fetch(`/benchmark/ref_movers_${ps}.u8`)).arrayBuffer());
    for(const mode of [14,4,34]){
      const m=masks[`${mode}@${+ps}`]; if(!m){out[`${mode}@${ps}`]='no mask';continue;}
      let sA=0,nA=0,sS=0,nS=0,sStat=0;
      for(let j=0;j<180;j++)for(let i=0;i<320;i++){const k=j*320+i;if(!(flg[k]&1)||!Number.isFinite(ref[k]))continue;
        const ao=m[(719-(2+4*j))*1280+2+4*i]/255,e=Math.abs(ao-ref[k]);sS+=e;nS++;
        if(flg[k]&2){sA+=e;nA++;sStat+=Math.abs(stat[k]-ref[k]);}}
      out[`${mode}@${ps}`]={maeAffected:+(sA/nA).toFixed(5),affectedPixels:nA,maeAllScored:+(sS/nS).toFixed(5),scoredPixels:nS,
        ignoreMoversMAE:+(sStat/nA).toFixed(5)};
    }
  }
  return out;
}
// AO error near moving objects for candidate moverScale vectors, against the ray-traced reference
// (export_reference_movers.py), on the pixels the movers change. Poses are split into a fit set and
// a held-out set, so a weight chosen on one is checked on the other.
//   await moverBench.scoreMoverScale([[1,1,1,1],[.5,.5,.5,.5]])
async function scoreMoverScale(scales,{fit=[0,7.4,14.8,22.2],hold=[3.7,11.1,18.5,25.9],mode=14}={}){
  const d=setup(), gl=d.renderer.getContext(), refs={};
  for(const t of [...fit,...hold]){const k=t.toFixed(1);
    refs[t]={ao:new Float32Array(await (await fetch(`/benchmark/ref_movers_${k}.f32`)).arrayBuffer()),
             flags:new Uint8Array(await (await fetch(`/benchmark/ref_movers_${k}.u8`)).arrayBuffer())};}
  d.setDynamicAO(true);
  const prev=d.dyn.moverScale.value.toArray();
  const score=poses=>{let sA=0,nA=0,sAll=0,nAll=0,signed=0;
    for(const [t,m] of poses){const {ao,flags}=refs[t];
      for(let j=0;j<180;j++)for(let i=0;i<320;i++){const k=j*320+i;if(!(flags[k]&1)||!Number.isFinite(ao[k]))continue;
        const e=m[(719-(2+4*j))*1280+2+4*i]/255-ao[k];sAll+=Math.abs(e);nAll++;if(flags[k]&2){sA+=Math.abs(e);signed+=e;nA++;}}}
    return {affected:+(sA/nA).toFixed(5),signed:+(signed/nA).toFixed(4),all:+(sAll/nAll).toFixed(5)};};
  const out=[];
  for(const s of scales){d.setMoverScale(...s);const masks={};
    for(const t of [...fit,...hold]){d.updateMovers(t);masks[t]=readMask(d,gl,mode);}
    out.push({scale:s,fit:score(fit.map(t=>[t,masks[t]])),hold:score(hold.map(t=>[t,masks[t]]))});}
  d.setMoverScale(...prev);
  return out;
}
window.moverBench={run,runTimes,ab,refMAE,scoreMoverScale,POSES};
})();
