from pathlib import Path
import json,hashlib,numpy as np
R=Path(__file__).resolve().parent;T=R.parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for layer in [0,2]:
 S=T/'fixtures-canonical'/f'c512-m64-l{layer}';m=json.loads((S/'probe-fixture.json').read_text());assert m['history']==128 and m['capacity']==512 and m['query_tokens']==64
 assert sha(S/'probe-input.f32')==m['input_sha256'];a=np.fromfile(S/'probe-input.f32',dtype='<f4')
 for rows,past in [(32,448),(64,128),(64,448)]:
  P=R/'fixtures'/f'l{layer}-m{rows}-p{past}';P.mkdir(parents=True,exist_ok=False);parts=[];audit=[];offset=0
  for x in m['input_contract']:
   n=int(np.prod(x['shape']));v=a[offset:offset+n].reshape(x['shape']);offset+=n;assert hashlib.sha256(v.tobytes()).hexdigest()==x['f32_sha256'];name=x['name']
   if '_past_' in name:
    assert not np.count_nonzero(v[:,:,128:,:]);z=v[:,:,:past,:].copy()
   elif '_mask' in name:
    assert np.all(v[:,:,:,128:512]==-10000)
    w=v.reshape(1,1,7,64,576)[:,:,:,:rows,:];z=np.concatenate([w[:,:,:,:,:past],w[:,:,:,:,512:512+rows]],axis=4).reshape(1,1,7*rows,past+rows)
   elif '_hidden' in name:z=v[:,:rows,:].copy()
   elif '_rope_' in name:z=v[:,:,:rows,:].copy()
   else:raise AssertionError(name)
   assert np.isfinite(z).all();parts.append(z.reshape(-1));audit.append({'name':name,'old_shape':x['shape'],'new_shape':list(z.shape),'sha256':hashlib.sha256(z.tobytes()).hexdigest()})
  assert offset==len(a);np.concatenate(parts).astype('<f4').tofile(P/'input.f32')
  d={'layer':layer,'history':128,'query':rows,'past_capacity':past,'source_meta_sha256':sha(S/'probe-fixture.json'),'input_sha256':sha(P/'input.f32'),'all_active_history_retained':True,'scope':'Canonical FP32 history128;query32 is first32 rows of canonicalquery64, not a two-chunk equivalence test','inputs':audit};(P/'meta.json').write_text(json.dumps(d,indent=2)+'\n')
