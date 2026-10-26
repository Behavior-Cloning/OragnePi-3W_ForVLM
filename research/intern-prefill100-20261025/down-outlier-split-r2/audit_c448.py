from pathlib import Path
import json,hashlib,re,sys
import numpy as np
R=Path(__file__).resolve().parent;rows=int(sys.argv[1]);D=R/('full-c448-m'+str(rows));S=R.parent/'c448-qualified-r3'/('original'+str(rows))
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
def weights(root):
 code=(root/'export/vnn_decoder.c').read_text();payload=(root/'export/decoder.export.data').read_bytes();out={}
 for label,body in re.findall(r'/\* (@[^*]+:weight) \*/(.*?)(?=/\*|$)',code,re.S):
  m=re.search(r'NEW_CONST_TENSOR\(const_tensor\[\d+\], attr, VSI_NN_TYPE_(\w+), (\d+), (\d+)\)',body);assert m,label
  typ,off,n=m.groups();raw=payload[int(off):int(off)+int(n)];fl=re.search(r'attr.dtype.fl = (-?\d+);',body)
  out[label]={'type':typ,'fl':int(fl.group(1)) if fl else None,'raw':raw,'sha256':hashlib.sha256(raw).hexdigest()}
 return out
original=weights(S);candidate=weights(D);data=np.load(D/'decoder.data',allow_pickle=True).item();checks=[]
for k,a in original.items():
 b=candidate[k]
 if k not in ['@b0_fullconnect_116:weight','@b1_fullconnect_116:weight']:
  assert a['type']==b['type'] and a['fl']==b['fl'] and a['raw']==b['raw'],k;checks.append({'name':k,'unchanged':True})
 else:
  assert b['type']=='INT16';name=k[1:].split(':')[0];actual=np.frombuffer(b['raw'],'<i2').reshape(896,4864).T
  expect=np.clip(np.rint(data[name]['weight']*2**b['fl']),-32768,32767).astype('<i2');assert np.array_equal(actual,expect),k
  checks.append({'name':k,'integer_quantization_of_split_source_verified':True,'fl':b['fl']})
for layer in [0,1]:
 k='@outlier_slowfc_%d:weight'%(8100+layer);v=candidate[k];assert v['type']=='FLOAT16'
 actual=np.frombuffer(v['raw'],'<f2').reshape(896,8).T;expect=data[k[1:].split(':')[0]]['weight'].astype('<f2');assert np.array_equal(actual,expect)
 checks.append({'name':k,'fp16_channels':8,'source_half_exact':True})
assert set(candidate)==set(original)|{'@outlier_slowfc_8100:weight','@outlier_slowfc_8101:weight'}
sys.path.insert(0,str(R.parent.parent.parent/'npu/kv-cache'));from fused_contract import verify_contract
ports=verify_contract(D,2);orig=verify_contract(S,2);assert ports['input_order']==orig['input_order'] and ports['output_order']==orig['output_order']
state=json.loads((D/'status.json').read_text());assert state['status']=='compiled-unvalidated' and sha(D/'export_nbg_unify/network_binary.nb')==state['nb_sha256']
out={'status':'static-checked-not-board-qualified','source_nb_sha256':sha(S/'export_nbg_unify/network_binary.nb'),'candidate_nb_sha256':state['nb_sha256'],'weights':checks,'ports':ports,'ports_source':'generated C; board actual port contract checked by probe','graph_sha256':sha(D/'decoder.json'),'quantizer_sha256':sha(D/'decoder.quantize'),'split_data_sha256':sha(D/'decoder.data')}
(R/('STATIC_C448_M'+str(rows)+'.json')).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
