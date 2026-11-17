from pathlib import Path
import json, subprocess, numpy as np
root=Path(__file__).resolve().parents[1]; out=root/'output/playwright/traced-run2'; out.mkdir(parents=True,exist_ok=True)
# Pin the old renderer so production changes cannot silently alter comparison baselines.
s=subprocess.check_output(['git','show','cfa1fd128ca3d9306079e564c2331c1a87535437:index.html'],cwd=root,text=True)
s=s.replace('uniform float tileRate;','uniform float tileRate;\nuniform float validationStart;\nuniform float validationFront;\nuniform float validationClamp;')
s=s.replace('float vis=0.;\n  for(int i=0;i<DIRS;i++){\n    float u=(float(i)+g1)/float(DIRS)', 'float nd=validationFront>.5?(frame<8.?4.:(frame<16.?3.:2.)):2.;\n  float vis=0.;\n  for(int i=0;i<4;i++){\n    if(float(i)>=nd)break;\n    float u=min((float(i)+g1)/nd,1.)')
s=s.replace('float cosn=dot(w,n),s=.02,v=1.;','float cosn=dot(w,n),s=validationStart,v=1.;')
s=s.replace('float e=min(.025,.1*s*cosn);','float e=max(min(.025,.1*s*cosn),validationClamp);')
s=s.replace('return vis/float(DIRS);','return vis/nd;')
s=s.replace('prevViewProj:traced.prevViewProj,frame:traced.frame,','validationStart:{value:.02},validationFront:{value:0},validationClamp:{value:0},prevViewProj:traced.prevViewProj,frame:traced.frame,')
s=s.replace('window.aoDemo={renderer,','window.aoDemo={scene,mesh,boxes,T,aoPass,field,renderer,')
s=s.replace('  renderer.setAnimationLoop(now=>{','  /* validation controls the render loop */\n  const validationLoop=(now)=>{')
s=s.replace('    frames=0;\n  });','    frames=0;\n  };')
s=s.replace('</script>\n</html>','</script>\n<script src="/benchmark/traced-validation.js"></script>\n</html>')
(out/'viewer.html').write_text(s)
views={'main':([18,-4,7.3],[10.8,8.8,1.4]),'west':([2.5,-3,5.5],[10,9,.8]),'north':([23,14.5,5],[11,5,1]),'low':([22,1.5,2.2],[9,11,1.2]),'inner':([5,13,3.2],[14,4,.8]),'above':([13,1,7.6],[12.5,9,0])}
boxes=np.array(json.loads((root/'benchmark/archive.json').read_text())); x,y=np.meshgrid(np.arange(2,1280,4),np.arange(2,720,4))
def unit(a):return a/np.linalg.norm(a,axis=-1,keepdims=True)
for name,(eye,target) in views.items():
 eye=np.array(eye); f=unit(np.array(target)-eye); right=unit(np.cross(f,[0,0,1])); up=np.cross(right,f)
 ray=unit(f+(x.ravel()[:,None]+.5-640)*2/720*np.tan(np.deg2rad(18.5))*right+(360-y.ravel()[:,None]-.5)*2/720*np.tan(np.deg2rad(18.5))*up)
 t=np.full(len(ray),np.inf); surface=np.full(len(ray),-1)
 for i,b in enumerate(boxes):
  with np.errstate(divide='ignore'): a=(b[:3]-b[3:]-eye)/ray; z=(b[:3]+b[3:]-eye)/ray
  near=np.minimum(a,z).max(-1); far=np.maximum(a,z).min(-1); hit=(near>=.1)&(far>=near)&(near<t); t[hit]=near[hit];surface[hit]=i
 valid=surface>=0; p=eye+ray[valid]*t[valid,None]; surf=surface[valid]; dist=np.full(len(p),np.inf)
 for i,b in enumerate(boxes):
  q=np.abs(p-b[:3])-b[3:]; d=np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(q.max(-1),0);d[surf==i]=np.inf; dist=np.minimum(dist,d)
 geom=np.load(root.parent/'thermite-kit-traced-ao/views'/f'{name}_geom.npz'); assert np.max(np.abs(p-geom['p']))<1e-4
 ref=np.load(root.parent/'thermite-kit-traced-ao/refs'/f'{name}_ao.npy')
 (out/f'{name}.json').write_text(json.dumps({'eye':eye.tolist(),'target':target,'indices':((719-y.ravel()[valid])*1280+x.ravel()[valid]).tolist(),'ref':ref.tolist(),'contact':(dist<=.05).astype(int).tolist(),'near12':(dist<=.12).astype(int).tolist(),'surface':surf.tolist()}))
print('Built isolated validation page and six aligned reference sets')
