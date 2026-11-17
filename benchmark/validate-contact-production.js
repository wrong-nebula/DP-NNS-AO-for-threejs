// Exercise the unmodified production shader; compare against saved GPU prototype pixels.
window.validateContactProduction=async()=>{
 const d=aoDemo,g=d.renderer.getContext();d.renderer.setAnimationLoop(null);d.controls.enableDamping=false;d.setDynamicAO(false);d.movers.forEach(m=>m.object.visible=false);d.setAOScale(.5);d.setMask(true);
 const results={};
 for(const name of ['main','west','north','low','inner','above']){
  const r=await(await fetch('/output/playwright/traced-run2/'+name+'.json')).json();
  const prototype=await(await fetch('/output/playwright/traced-run2/astra-contact-grid1-'+name+'-masks.json')).json();
  d.camera.position.fromArray(r.eye);d.controls.target.fromArray(r.target);d.controls.update();d.camera.lookAt(...r.target);d.camera.updateMatrixWorld();d.traced.frame.value=0;d.resetHistory();
  const frames={},masks={};
  for(let f=1;f<=64;f++){
   d.render(64);
   if([1,4,16,64].includes(f)){
    const pixels=new Uint8Array(1280*720*4);g.readPixels(0,0,1280,720,g.RGBA,g.UNSIGNED_BYTE,pixels);const a=r.indices.map(i=>pixels[i*4]/255);masks[f]=a;
    let error=0,contact=0,light=0,count=0,maxDelta=0,differing=0;
    for(let i=0;i<a.length;i++){let e=a[i]-r.ref[i];error+=Math.abs(e);if(r.contact[i]){count++;contact+=Math.abs(e);light+=Math.max(0,e);}const delta=Math.abs(a[i]-prototype.next[f][i]);maxDelta=Math.max(maxDelta,delta);if(delta>0)differing++;}
    frames[f]={mae:error/a.length,contactMAE:contact/count,contactTooLight:light/count,maxPrototypeDelta:maxDelta,differingSamples:differing};
   }
   if(f%8===0)await new Promise(ok=>setTimeout(ok,0));
  }
  results[name]=frames;
  await fetch('/save/astra-contact-production-'+name+'-masks.json',{method:'POST',body:JSON.stringify(masks)});
 }
 await fetch('/save/astra-contact-production-results.json',{method:'POST',body:JSON.stringify(results,null,2)});return results;
};
