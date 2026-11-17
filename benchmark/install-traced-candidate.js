// Install an independently reviewed GLSL port without changing the shipped viewer.
// `glsl` defines candidateAO(vec3 p, vec3 n); it may call staticDistance/ign and uses frame.
// Legacy seed/champion variants remain available in the same GPU program.
window.installTracedCandidate=({glsl,blendK,tag})=>{
 const d=aoDemo,m=d.aoPass.material;
 if(!window.originalValidationFragment)window.originalValidationFragment=m.fragmentShader;
 const source=window.originalValidationFragment.replace('float tracedAO(vec3 p,vec3 n){','float legacyTracedAO(vec3 p,vec3 n){');
 const marker='// The previous frame';
 if(!source.includes(marker)||!glsl.includes('candidateAO'))throw Error('Invalid candidate source');
 m.fragmentShader=source.replace(marker,glsl+'\nfloat tracedAO(vec3 p,vec3 n){return validationFront>1.5?candidateAO(p,n):legacyTracedAO(p,n);}\n'+marker);
 m.needsUpdate=true;
 gpuValidation.variants.next=[.12,2,1e-6,blendK];
 window.validationTag=tag;
 return {tag,blendK};
};
