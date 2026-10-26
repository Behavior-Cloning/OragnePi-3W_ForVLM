from pathlib import Path
import json,hashlib,numpy as np
R=Path(__file__).resolve().parent;T=R.parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for tokens,past in [(32,448),(32,256),(64,128),(64,256)]:
 S=T/'fixtures-canonical'/f'c512-m{tokens}-l0';P=R/'fixtures'/f'm{tokens}-p{past}';P.mkdir(parents=True,exist_ok=False);m=json.loads((S/'probe-fixture.json').read_text());assert m['history']==128 and m['capacity']==512 and m['query_tokens']==tokens
 assert sha(S/'probe-input.f32')==m['input_sha256'];a=np.fromfile(S/'probe-input.f32',dtype='<f4');parts=[];audit=[];offset=0
 for x in m['input_contract']:
  n=int(np.prod(x['shape']));v=a[offset:offset+n].reshape(x['shape']);offset+=n;assert hashlib.sha256(v.tobytes()).hexdigest()==x['f32_sha256'];name=x['name'];z=v.copy()
  if '_past_' in name:
   assert not np.count_nonzero(v[:,:,128:,:]);z=v[:,:,:past,:].copy()
  elif '_mask' in name:
   assert np.all(v[:,:,:,128:512]==-10000);z=np.concatenate([v[:,:,:,:past],v[:,:,:,512:]],axis=3)
  assert np.isfinite(z).all();parts.append(z.reshape(-1));audit.append({'name':name,'old_shape':x['shape'],'new_shape':list(z.shape),'sha256':hashlib.sha256(z.tobytes()).hexdigest()})
 assert offset==len(a);np.concatenate(parts).astype('<f4').tofile(P/'input.f32');d={'history':128,'query':tokens,'past_capacity':past,'source_meta_sha256':sha(S/'probe-fixture.json'),'input_sha256':sha(P/'input.f32'),'all_active_history_retained':True,'hidden_and_rope_unchanged':True,'scope':'FP32 canonical history128, not real NPU runtime KV or longer256history','inputs':audit};(P/'meta.json').write_text(json.dumps(d,indent=2)+'\n');print(tokens,past,d['input_sha256'])
