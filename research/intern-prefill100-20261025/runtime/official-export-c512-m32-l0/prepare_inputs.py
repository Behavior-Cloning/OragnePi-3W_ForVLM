from pathlib import Path
import json,hashlib,numpy as np
P=Path(__file__).resolve().parent;T=P.parent.parent;S=T/'fixtures-canonical/c512-m32-l0'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=json.loads((S/'probe-fixture.json').read_text());d=np.fromfile(S/'probe-input.f32',dtype='<f4');assert sha(S/'probe-input.f32')==f['input_sha256']
p=json.loads((P/'source-proof.json').read_text());off=0
for i,((name,n),contract,item) in enumerate(zip(f['inputs'],f['input_contract'],p['inputs'])):
 x=d[off:off+n];assert np.isfinite(x).all() and len(x)==item['elements'] and name==contract['name'] and item['name'].startswith(name+'_')
 assert hashlib.sha256(x.tobytes()).hexdigest()==contract['f32_sha256']
 for v in ['plain','bitcast']:np.savetxt(P/v/item['file'],x,fmt='%.9g')
 y=np.loadtxt(P/'plain'/item['file'],dtype=np.float32);assert np.array_equal(x,y)
 item['sha256']=sha(P/'plain'/item['file']);item['bytes']=(P/'plain'/item['file']).stat().st_size;item['source_f32_sha256']=contract['f32_sha256'];item['float32_roundtrip_exact']=True
 assert sha(P/'bitcast'/item['file'])==item['sha256'];off+=n
assert off==len(d)
p['input_sources']=str(S/'probe-input.f32');p['canonical_input_sha256']=f['input_sha256'];p['input_definition']='Actual frozen Chinese canonical FP32 layer-state inputs, not SDK generated random tensors';p['source_nb_sha256']=f['nb_sha256']
(P/'source-proof.json').write_text(json.dumps(p,indent=2)+'\n');print('actual canonical11 ready',f['input_sha256'])
