#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,numpy as np
T=Path(__file__).resolve().parent;src=T/'fixtures-canonical/c512-m256-l0';dst=T/'fixtures-compact/c128-m256-l0';dst.mkdir(parents=True,exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((src/'probe-fixture.json').read_text());assert m['history']==128 and m['capacity']==512 and m['query_tokens']==256
assert sha(src/'probe-input.f32')==m['input_sha256'];assert sha(src/'probe-reference.f32')==m['reference_sha256']
a=np.fromfile(src/'probe-input.f32',dtype='<f4');offset=0;parts=[];audit=[]
for x in m['input_contract']:
 shape=x['shape'];n=int(np.prod(shape));v=a[offset:offset+n].reshape(shape);offset+=n
 assert hashlib.sha256(v.tobytes()).hexdigest()==x['f32_sha256'];name=x['name'];out=v.copy();op='unchanged byte-exact'
 if '_past_' in name:
  assert not np.count_nonzero(v[:,:,128:,:]);out=v[:,:,:128,:].copy();op='keep all128 active past positions, remove384 all-zero inactive positions'
 elif '_mask' in name:
  assert np.all(v[:,:,:,128:512]==-10000.0);out=np.concatenate([v[:,:,:,:128],v[:,:,:,512:]],axis=3);op='preserve active128 past +256 query key columns; remove only masked past columns'
 assert np.isfinite(out).all();parts.append(out.reshape(-1));audit.append({'name':name,'old_shape':shape,'new_shape':list(out.shape),'elements':int(out.size),'source_f32_sha256':x['f32_sha256'],'f32_sha256':hashlib.sha256(out.tobytes()).hexdigest(),'operation':op})
assert offset==len(a)
np.concatenate(parts).astype('<f4').tofile(dst/'probe-input.f32');(dst/'probe-reference.f32').symlink_to((src/'probe-reference.f32').resolve())
result={'first_layer':0,'fusion_width':2,'history':128,'capacity':128,'query_tokens':256,'source_meta_sha256':sha(src/'probe-fixture.json'),'source_input_sha256':m['input_sha256'],'input_sha256':sha(dst/'probe-input.f32'),'reference_sha256':m['reference_sha256'],'same_hidden_and_rope':True,'same_all_active_KV':True,'same_all_active_mask_columns':True,'outputs':m['outputs'],'input_contract':audit,'token_ids':m['token_ids'],'scope':'repacked canonical FP32 history128 fixture; not captured NPU KV, no runtime dispatch changes'}
(dst/'probe-fixture.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'input_bytes':(dst/'probe-input.f32').stat().st_size,'input_sha256':result['input_sha256'],'history':128,'query_tokens':256}))
