"""Offline actual exported dense-weight shape accounting; no latency/roofline claim."""
from pathlib import Path
import json,hashlib,numpy as np
R=Path(__file__).resolve().parents[2];BASE=Path('/opt/opi-lab/source-validation-20261017/gguf-fresh/export/w8-fused2-c1280-t16-cap1-productwide');out=R/'results/dense-work-budget';out.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
matrices=[];sources=[]
for first in range(0,24,2):
 p=BASE/f'l{first}';g=json.loads((p/'decoder.json').read_text());data=np.load(p/'decoder.data',allow_pickle=True).item();nodes=[(n,v) for n,v in g['Layers'].items() if v['op']=='fullconnect'];assert len(nodes)==14
 sources.append({'first_layer':first,'graph_sha256':sha(p/'decoder.json'),'data_sha256':sha(p/'decoder.data')})
 for name,v in nodes:
  w=data[name]['weight'];assert w.ndim==2 and v['parameters']['weights'] in w.shape
  layer=first+int(name[1]);assert layer in [first,first+1]
  matrices.append({'layer':layer,'node':name,'shape':list(w.shape),'dense_MAC_per_token':int(w.size)})
 del data
assert len(matrices)==168
per_token=sum(m['dense_MAC_per_token'] for m in matrices)
cells=[]
for static in [64,128]:
 for body in [64,128,192,256]:
  n=body+8;mac=per_token*n
  cells.append({'static_total_tokens':static,'system_body_tokens':static-5,'system_template_tokens':5,'dynamic_body_tokens':body,'dynamic_template_tokens':8,'new_tokens':n,'dense_GMAC':mac/1e9,'dense_effective_TOPS_needed_for_100ms':mac*2/.1/1e12})
a={'scope':'24-layer dense projections only, direct weight-shape count; multiply+add counted as two operations. Excludes attention QK/AV, norms, rotary, nonlinearities, head, transfers and host setup. Not actual runtime instructions, chip peak throughput or latency lower-bound. Quantization does not reduce logical dense MAC count. No claim goal impossible or achieved.','dense_MAC_per_new_token':per_token,'matrices':matrices,'source_sha256':sources,'cells':cells,'script_sha256':sha(Path(__file__))}
(out/'ANALYSIS.json').write_text(json.dumps(a,indent=2)+'\n')
t='# Dense arithmetic budget for the 100ms objective\n\nActual 24-layer exported dense matrices, 168 projections, '+str(per_token)+' MAC per new token. Counts only dense mathematical operations, not measured engine work or a wall-time bound. No board access.\n\nSystem total64/128=body59/123+template5, only static KV reused. Fixed length does not change dense work per newly computed token; attention work can change.\n\n|Dynamic body|New incl template|Dense GMAC|Required dense-only effective TOPS within100ms|\n|---:|---:|---:|---:|\n'
for c in cells[:4]:t+=f"|{c['dynamic_body_tokens']}|{c['new_tokens']}|{c['dense_GMAC']:.3f}|{c['dense_effective_TOPS_needed_for_100ms']:.3f}|\n"
t+='\nMultiply and add count separately. Attention, normalization, rotary, activations, vocabulary head, transfers and setup need additional time/work. Do not compare this directly with a vendor TOPS number without matching datatype and sustained utilization. This does not establish either feasibility or impossibility of the target.\n'
(out/'REPORT.md').write_text(t);print(t)
