from pathlib import Path
import numpy as np,json,hashlib
T=Path(__file__).resolve().parents[2];D=T/'fixtures-canonical/c512-m256-l0';m=json.loads((D/'probe-fixture.json').read_text());raw=(D/'probe-input.f32').read_bytes();assert hashlib.sha256(raw).hexdigest()==m['input_sha256'];v=np.frombuffer(raw,dtype='<f4');off=0;rows=[]
for part in m['input_contract']:
 shape=part['shape'];n=int(np.prod(shape));a=v[off:off+n].reshape(shape);off+=n;name=part['name'];assert hashlib.sha256(a.tobytes()).hexdigest()==part['f32_sha256']
 if '_past_' in name:
  pad=a[:,:,128:,:];assert np.count_nonzero(pad)==0;rows.append({'name':name,'past_shape':shape,'active_history':128,'empty_past_slots':384,'empty_slots_all_zero':True})
 if '_mask' in name:
  pad=a[:,:,:,128:512];assert np.all(pad==-10000.0);rows.append({'name':name,'mask_shape':shape,'discardable_columns':[128,512],'all_discardable_mask_values':-10000.0,'keep_key_columns':[[0,128],[512,768]],'preserved_new_columns':256})
assert off==len(v)
result={'scope':'host validation of one canonical history128 fixture only; not yet a specialized graph or runtime implementation','input_sha256':m['input_sha256'],'rows':rows,'proposed_physical_keys':384,'current_physical_keys':768,'semantic_history_retained':128,'queries_retained':256,'requires_runtime_checks':['only dispatch if all active past positions fit128','retain RoPE absolute positions','remap mask columns alongside K/V','subsequent chunks with past>128 must use adequate history shape','end-to-end numeric and output parity still required']}
(T/'runtime/profile-analysis/empty-past-slots-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
