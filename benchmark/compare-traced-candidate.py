"""Summarize an independently GPU-ported candidate; never adopts a candidate itself."""
from pathlib import Path
import json,sys,numpy as np
p=Path(__file__).resolve().parents[1]/'output/playwright/traced-run2';tag=sys.argv[1]
vs=['main','west','north','low','inner','above'];r={v:json.loads((p/f'{tag}-{v}-results.json').read_text()) for v in vs}
summary={'tag':tag,'view_error':{},'contact':{},'per_view':{}}
for v in vs:
 summary['per_view'][v]={f:{k:r[v][k][f]['all']['mae'] for k in ['seed','champion','next']} for f in ['1','4','16','64']}
for f in ['1','4','16','64']:
 summary['view_error'][f]={g:{k:float(np.mean([r[v][k][f]['all']['mae'] for v in views])) for k in ['seed','champion','next']} for g,views in [('fitting',vs[:3]),('held_out',vs[3:])]}
for f in ['16','64']:
 n=sum(r[v]['seed'][f]['contact5cm']['n'] for v in vs)
 summary['contact'][f]={'n':n,'metrics':{k:{m:sum(r[v][k][f]['contact5cm'][m]*r[v][k][f]['contact5cm']['n'] for v in vs)/n for m in ['mae','underOcclusion','overOcclusion']} for k in ['seed','champion','next']}}
 counters={k:0 for k in ['worse_than_seed_4of255','worse_than_champion_4of255','accurate_seed_contacts','accurate_seed_worse_4of255']};worst=[]
 for v in vs:
  refs=json.loads((p/f'{v}.json').read_text());masks=json.loads((p/f'{tag}-{v}-masks.json').read_text());mask=np.array(refs['contact'],bool);ref=np.array(refs['ref']);se=np.abs(np.array(masks['seed'][f])-ref);ce=np.abs(np.array(masks['champion'][f])-ref);ne=np.abs(np.array(masks['next'][f])-ref)
  counters['worse_than_seed_4of255']+=int((mask&(ne>se+4/255)).sum());counters['worse_than_champion_4of255']+=int((mask&(ne>ce+4/255)).sum());acc=mask&(se<=.03);counters['accurate_seed_contacts']+=int(acc.sum());counters['accurate_seed_worse_4of255']+=int((acc&(ne>se+4/255)).sum())
  ids=np.flatnonzero(mask);ids=ids[np.argsort((ne-se)[ids])[-5:][::-1]]
  worst.extend({'view':v,'pixel':refs['indices'][int(i)],'reference':float(ref[i]),'seed':masks['seed'][f][int(i)],'champion':masks['champion'][f][int(i)],'next':masks['next'][f][int(i)]} for i in ids)
 summary['contact'][f].update(counters);summary['contact'][f]['worst_examples']=worst
(p/f'{tag}-comparison.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:val for k,val in summary.items() if k!='per_view'},indent=2))
