#!/usr/bin/env python3
"""Inspect generated C/constants from rejected all-W8 Conv candidate; never compile or board access."""
from pathlib import Path
import re,json,hashlib,numpy as np
R=Path(__file__).resolve().parent;SRC=R.parent/'cases/c512-m256-l0';OUT=R/'conv1x1-affine-c512-m256-l0'
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
def numerical(sig):
 d={'dtype':sig['dtype']}
 if 'scales_sha256' in sig:
  d.update({k:v for k,v in sig.items() if k in ['scales_sha256','zero_points_sha256','scale_dim','zero_points_dim']})
 else:
  d['scale_f32']=float(np.float32(2.**(-int(sig['fl'])))) if 'fl' in sig else float(np.float32(float(sig['scale'].rstrip('f'))))
  d['zero_point']=int(sig.get('zero_point','0'))
 return d
def tensor(folder,code,b):
 m=re.search(r'VSI_NN_TYPE_(\w+),\s*(\d+),\s*(\d+)\)',b);typ,off,num=m.groups();data=(folder/'export/decoder.export.data').read_bytes()[int(off):int(off)+int(num)]
 return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'signature':sig(code,b)}
a=(SRC/'export/vnn_decoder.c').read_text();b=(OUT/'export/vnn_decoder.c').read_text();ab=blocks(a);bb=blocks(b);g=json.loads((SRC/'decoder.json').read_text());records=[]
for k,node in g['Layers'].items():
 if node['op']!='fullconnect':continue
 rec={'key':k}
 for f in ['weight','bias']:
  x=tensor(SRC,a,ab['@'+k+':'+f]);y=tensor(OUT,b,bb['@'+k+':'+f]);same=x['bytes']==y['bytes'] and x['sha256']==y['sha256'] and numerical(x['signature'])==numerical(y['signature']);rec[f]={'bytes_dtype_and_numerical_grid_same':same,'source':x,'candidate':y};assert same,(k,f,x,y)
 for p in ['@'+k+':out0',node['inputs'][0]]:
  x=sig(a,ab[p]);y=sig(b,bb[p]);assert numerical(x)==numerical(y),(k,p,x,y);assert x['dtype']==y['dtype']=='INT16';rec[p]={'numerical_grid_same':True,'source_signature':x,'candidate_signature':y}
 records.append(rec)
assert len(records)==14
quant_same=sha(SRC/'decoder.quantize')==sha(OUT/'decoder.quantize');assert not quant_same
node=re.search(r'(?s)/\* ([^*]+) \*/\s*node\[22\] =[^;]+;',b)
manifest={'state':'generated_c_audited_ready_for_bitcast','reason':'Expected SDK DFP I16 to ASYM I16 DATACONVERT rejection; generated C and payload prepared for separate BITCAST step, no NB','quantizer_byte_equal':quant_same,'all14_weights_biases_dtype_bytes_and_scale_equal':True,'all14_input_output_I16_numerical_grid_equal':True,'records':records,'candidate_source_json_sha256':sha(OUT/'decoder.json'),'candidate_source_data_sha256':sha(OUT/'decoder.data'),'candidate_quantizer_sha256':sha(OUT/'decoder.quantize'),'export_log_sha256':sha(OUT/'export.log'),'compiler_c_sha256':sha(OUT/'export/vnn_decoder.c'),'compiler_data_sha256':sha(OUT/'export/decoder.export.data'),'binary_exists':(OUT/'export_nbg_unify/network_binary.nb').exists(),'source_binary_sha256':sha(SRC/'export_nbg_unify/network_binary.nb'),'board_used':False}
(R/'m256-affine-generated-audit.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in manifest.items() if k!='records'},indent=2))
for line in b.splitlines():
 if 'node[16]' in line:print(line)
