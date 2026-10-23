from pathlib import Path
import torch,numpy as np,json,time,hashlib
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel,Qwen3_5TextRotaryEmbedding
from model_layers import MODEL,CFG,ROOT
r=Path(__file__).resolve().parents[1];torch.set_num_threads(4);torch.set_grad_enabled(False);start=time.time()
with safe_open(MODEL/'model.safetensors',framework='pt') as f:state={k[len('model.language_model.'):]:f.get_tensor(k).float() for k in f.keys() if k.startswith('model.language_model.')}
with torch.device('meta'):hf=Qwen3_5TextModel(CFG)
hf.load_state_dict(state,assign=True,strict=True);del state;hf.rotary_emb=Qwen3_5TextRotaryEmbedding(CFG);hf.eval();peaks={};records=[]
cases=json.loads((r/'results/w8-household-calibration.json').read_text())['texts']
for case in cases:
 cache=None;ids=case['prompt_ids']+case['calibration_generated_ids'][:4]
 for j,token in enumerate(ids):
  h=hf(torch.tensor([[token]]),past_key_values=cache,use_cache=True);cache=h.past_key_values
  for i,l in enumerate(cache.layers):
   if getattr(l,'recurrent_states',None) is not None:
    p=l.recurrent_states.abs().amax(dim=(-2,-1)).flatten().numpy()
    peaks[str(i)]=np.maximum(peaks.get(str(i),np.zeros(16)),p)
 records.append({'id':case['id'],'tokens':len(ids)});print('case',case['id'],len(ids),flush=True)
result={'scope':'8 calibration prompts, all token states, no heldout input','cases':records,'peaks':{k:v.tolist() for k,v in peaks.items()},'scales':{k:(2**np.ceil(np.log2(np.maximum(v,1e-4)))).tolist() for k,v in peaks.items()},'i16_fl':14,'headroom':'normalized peak <=1, representable range about +/-2','seconds':time.time()-start}
(r/'results/w8-state-head-calibration.json').write_text(json.dumps(result,indent=2));print('COMPLETE',time.time()-start,flush=True)
