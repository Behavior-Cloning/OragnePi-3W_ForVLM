#!/usr/bin/env python3
"""Inspect generated C/constants from rejected all-W8 Conv candidate; never compile or board access."""
from pathlib import Path
import re,json,hashlib,numpy as np
R=Path(__file__).resolve().parent;SRC=R.parent/'cases/c512-m32-l0';OUT=R/'conv1x1-c512-m32-l0'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def blocks(code):return {key:body for names,body in re.findall(r'/\* (.*?)\*/(.*?)(?=/\*|$)',code,re.S) for key in names.split() if key.startswith('@')}
def sig(code,b):
 d={k:v for k,v in re.findall(r'attr.dtype.(\w+) = ([^;]+);',b) if k not in ['channel_dim','scales','zero_points']}
 typ=re.findall(r'VSI_NN_TYPE_(\w+)',b);d['dtype']=typ[-1]
 for fld in ['scales','zero_points']:
  m=re.search(r'attr.dtype.'+fld+r' = (\w+);',b)
  if m:
   a=re.search(r'\b'+m[1]+r'\[\]\s*=\s*\{([^}]+)\}',code);v=np.asarray([float(x.strip().rstrip('f')) for x in a[1].split(',') if x.strip()],dtype='<f4' if fld=='scales' else '<i4');d[fld+'_sha256']=hashlib.sha256(v.tobytes()).hexdigest()
 return d
def tensor(folder,code,b):
 m=re.search(r'VSI_NN_TYPE_(\w+),\s*(\d+),\s*(\d+)\)',b);typ,off,num=m.groups();data=(folder/'export/decoder.export.data').read_bytes()[int(off):int(off)+int(num)]
 return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'signature':sig(code,b)}
a=(SRC/'export/vnn_decoder.c').read_text();b=(OUT/'export/vnn_decoder.c').read_text();ab=blocks(a);bb=blocks(b);g=json.loads((SRC/'decoder.json').read_text());records=[]
for k,node in g['Layers'].items():
 if node['op']!='fullconnect':continue
 rec={'key':k}
 for f in ['weight','bias']:
  x=tensor(SRC,a,ab['@'+k+':'+f]);y=tensor(OUT,b,bb['@'+k+':'+f]);rec[f]={'same':x==y,'source':x,'candidate':y};assert x==y,(k,f,x,y)
 for p in ['@'+k+':out0',node['inputs'][0]]:
  x=sig(a,ab[p]);y=sig(b,bb[p]);assert x==y,(k,p,x,y);assert x['dtype']=='INT16';rec[p]={'quantization_same':True,'signature':x}
 records.append(rec)
assert len(records)==14
quant_same=sha(SRC/'decoder.quantize')==sha(OUT/'decoder.quantize');assert quant_same
node=re.search(r'(?s)/\* ([^*]+) \*/\s*node\[22\] =[^;]+;',b)
manifest={'state':'rejected_no_binary','reason':'SDK CONV2D op_check rejects DFP INT16 activation plus DFP INT8 weight at node22','quantizer_byte_equal':quant_same,'all14_weights_biases_dtype_bytes_and_scale_equal':True,'all14_input_output_I16_quant_equal':True,'records':records,'candidate_source_json_sha256':sha(OUT/'decoder.json'),'candidate_source_data_sha256':sha(OUT/'decoder.data'),'candidate_quantizer_sha256':sha(OUT/'decoder.quantize'),'export_log_sha256':sha(OUT/'export.log'),'compiler_c_sha256':sha(OUT/'export/vnn_decoder.c'),'compiler_data_sha256':sha(OUT/'export/decoder.export.data'),'binary_exists':(OUT/'export_nbg_unify/network_binary.nb').exists(),'source_binary_sha256':sha(SRC/'export_nbg_unify/network_binary.nb'),'board_used':False}
(R/'rejected-candidate-audit.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in manifest.items() if k!='records'},indent=2))
for line in b.splitlines():
 if 'node[22]' in line:print(line)
