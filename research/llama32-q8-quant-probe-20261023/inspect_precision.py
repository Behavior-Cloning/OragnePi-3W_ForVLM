"""Audit retained actual compiler graph/quantization files (not requested flags)."""
from pathlib import Path
import json,yaml,hashlib,collections
R=Path(__file__).resolve().parent;result={}
for name,d in [(x,R/'evidence'/x) for x in ['layer0-allw8','layer0-mixed','head0-w8','head1-w8','head2-w8','head3-w8']]+[('q4-layer0-allw8',R/'q4-control')]:
 g=json.loads((d/'decoder.json').read_text())['Layers'];q=yaml.safe_load((d/'decoder.quantize').read_text())['quantize_parameters'];conv=json.loads((d/'conversion.json').read_text());mat=[]
 def small(v):
  v=dict(v)
  if 'scale' in v:
   scales=v.pop('scale');v['scale_count']=len(scales);v['scale_min']=min(scales);v['scale_max']=max(scales)
  return v
 for k,v in g.items():
  if v['op']!='fullconnect':continue
  entry=next((x for x in conv.get('weights',[]) if x['name']==k),{})
  mat.append(dict(node=k,name=v['name'],shape=entry.get('shape'),weight=small(q['@'+k+':weight']),inputs=[dict(node=x,quant=small(q.get(x,{}))) for x in v['inputs']],output=small(q.get('@'+k+':out0',{}))))
 io=[dict(node=k,name=v['name'],op=v['op'],quant=small(q.get('@'+k+':out0',{}))) for k,v in g.items() if v['op'] in ('input','output')]
 floating=[dict(node=k,name=g.get(k[1:].split(':')[0],{}).get('name'),op=g.get(k[1:].split(':')[0],{}).get('op'),qtype=v['qtype']) for k,v in q.items() if v.get('qtype')=='float16' and not k.endswith((':weight',':bias'))]
 result[name]=dict(quantize_sha256=hashlib.file_digest((d/'decoder.quantize').open('rb'),'sha256').hexdigest(),graph_sha256=hashlib.file_digest((d/'decoder.json').open('rb'),'sha256').hexdigest(),fc_count=len(mat),fc_weight_types=dict(collections.Counter(x['weight']['qtype'] for x in mat)),fc=mat,io=io,activation_types=dict(collections.Counter(v.get('qtype') for k,v in q.items() if k.endswith(':out0'))),fp16_activations=floating,norm_softmax=[dict(node=k,name=v['name'],op=v['op'],quant=small(q.get('@'+k+':out0',{}))) for k,v in g.items() if v['op'] in ('softmax','sqrt','reduce')],note='From retained compiler input graph and actual quantize file; NBG external IO verified separately via vip_query logs. Generated C was not retained in the first bounded build, so this is not an internal NBG disassembly.')
(R/'precision-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:{a:b for a,b in v.items() if a in ['fc_count','fc_weight_types','activation_types','io']} for k,v in result.items()},indent=2))
