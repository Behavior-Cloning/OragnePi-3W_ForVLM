#!/usr/bin/env python3
"""Seven actual Gemma2 FC weights int8, i16 FC IO, nonlinear ops stay FP16."""
from pathlib import Path
import json,yaml,numpy as np,hashlib,math,os,shutil
r=Path(__file__).resolve().parent;b=Path('/opt/opi-lab/paligemma2-npu-compat-20261023');src=b/'fp16';dst=b/'w8';dst.mkdir(exist_ok=True)
for n in ['decoder.json','decoder.quantize','tensor-inputmeta.yml','dataset.txt']:shutil.copyfile(src/n,dst/n)
if not (dst/'decoder.data').exists():os.link(src/'decoder.data',dst/'decoder.data')
s=json.loads((r/'export-summary.json').read_text());lookup={v:k for k,v in s['import_weight_sha256'].items()};data=np.load(src/'decoder.data',allow_pickle=True).item()
g=json.loads((dst/'decoder.json').read_text());q=yaml.safe_load((dst/'decoder.quantize').read_text());qp=q['quantize_parameters'];audit=[]
def dfp(am):
 am=max(am*2,1e-6);fl=max(-31,min(31,math.floor(math.log2(32767/am))))
 return dict(qtype='i16',quantizer='dynamic_fixed_point',fl=fl,max=am,min=-am)
for name,node in list(g['Layers'].items()):
 if node['op']!='fullconnect':continue
 w=data[name]['weight'];h=hashlib.sha256(w.tobytes()).hexdigest();module=lookup[h];ir,orr=s['calibration_ranges'][module];scale=np.maximum(np.max(np.abs(w),axis=0)/127.,1e-12)
 qp['@'+name+':weight']=dict(qtype='i8',quantizer='perchannel_symmetric_affine',rounding='rtne',scale=scale.tolist(),channel_dim=-1)
 conv='fc_input_convert_'+name;g['Layers'][conv]=dict(name=conv,op='dtype_converter',parameters={},inputs=[node['inputs'][0]],outputs=['out0']);node['inputs'][0]='@'+conv+':out0'
 qp['@'+conv+':out0']=dfp(ir);qp['@'+name+':out0']=dfp(orr)
 qp['@'+name+':bias']=dict(qtype='i64',quantizer='dynamic_fixed_point',fl=24)
 audit.append(dict(node=name,module=module,weight_sha256=h,shape=list(w.shape),input_quant=qp['@'+conv+':out0'],output_quant=qp['@'+name+':out0']))
assert len(audit)==7
(dst/'decoder.json').write_text(json.dumps(g,indent=2));(dst/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False));(r/'w8-fc-audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
