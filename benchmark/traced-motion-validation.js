window.runMotionValidation=async({variants=['seed','champion']}={})=>{
 const V=gpuValidation,d=aoDemo,g=d.renderer.getContext(),r=V.refs.main,out={motion:{},movers:{},timings:Object.fromEntries(variants.map(v=>[v,{early:[],orbit:[]}]))};
 function orbit(angle){let x=r.eye[0]-r.target[0],y=r.eye[1]-r.target[1];d.camera.position.set(r.target[0]+x*Math.cos(angle)-y*Math.sin(angle),r.target[1]+x*Math.sin(angle)+y*Math.cos(angle),r.eye[2]);d.camera.lookAt(...r.target);d.camera.updateMatrixWorld();}
 function metric(ref,mask){let p=V.capture(d),n=0,err=0;for(let i=0;i<r.indices.length;i++){if((mask&&!mask[i])||ref[i]===null)continue;n++;err+=Math.abs(p[r.indices[i]*4]/255-ref[i]);}return {n,mae:err/n};}
 for(const v of variants){out.motion[v]={};for(const speed of [1,3,10]){V.setup(v,r);for(let i=0;i<64;i++){orbit((i-63)*speed*Math.PI/180);d.render(64);}out.motion[v][speed]=metric(r.ref);await new Promise(ok=>setTimeout(ok,0));}}
 const refs=await(await fetch('/output/playwright/traced-run2/movers.json')).json();
 for(const v of variants){out.movers[v]={};for(const [t,ref] of Object.entries(refs)){V.setup(v,r);d.movers.forEach(m=>m.object.visible=true);d.setDynamicAO(true);d.updateMovers(+t);for(let i=0;i<64;i++)d.render(64);out.movers[v][t]={all:metric(ref.ao,ref.scored),near:metric(ref.ao,ref.affected.map((a,i)=>a&&ref.scored[i]))};await new Promise(ok=>setTimeout(ok,0));}}
 // Interleaved readback-closed wall time per frame; exclude warm-up shader compilation.
 for(let block=0;block<12;block++)for(const v of block%2?[...variants].reverse():variants){V.setup(v,r);for(let i=0;i<16;i++)d.render(64);V.capture(d);d.resetHistory();d.traced.frame.value=0;V.capture(d);let begin=performance.now();for(let i=0;i<16;i++)d.render(64);V.capture(d);out.timings[v].early.push((performance.now()-begin)/16);
 for(let i=0;i<64;i++){orbit(i*Math.PI/180);d.render(64);}V.capture(d);begin=performance.now();for(let i=64;i<80;i++){orbit(i*Math.PI/180);d.render(64);}V.capture(d);out.timings[v].orbit.push((performance.now()-begin)/16);await new Promise(ok=>setTimeout(ok,0));}
 await V.save('readback-timing-results.json',Object.fromEntries(variants.map(v=>[v,out.timings[v].early])));await V.save('motion-timing-results.json',out);return out;
};
