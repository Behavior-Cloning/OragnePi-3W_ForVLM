"""Compare fresh ONNX-export fixtures with independently mapped Transformers."""
import argparse
import json
from pathlib import Path
import gguf
import numpy as np
import torch
from validate_standard_oracle import load_standard, errors

p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
a=p.parse_args();torch.set_num_threads(2)
model,config=load_standard(gguf.GGUFReader(a.model));records=[]
with torch.inference_mode():
    for case in (0,1):
        ids=json.loads((a.source/f'case{case}-token-ids.json').read_text())
        standard=model(input_ids=torch.tensor([ids]),use_cache=False).logits
        actual=torch.from_numpy(np.fromfile(a.source/f'case{case}-logits.f32',np.float32).reshape(standard.shape))
        result=errors(actual,standard);assert result['relative_l2']<=1e-4,result
        matches=int((actual.argmax(-1)==standard.argmax(-1)).sum());assert matches==len(ids)
        # Independent persistent-cache path: prefill then scalar decode.
        first=model(input_ids=torch.tensor([ids[:8]]),use_cache=True)
        cache=first.past_key_values;pieces=[first.logits]
        for token in ids[8:]:
            step=model(input_ids=torch.tensor([[token]]),past_key_values=cache,use_cache=True)
            cache=step.past_key_values;pieces.append(step.logits)
        cached=torch.cat(pieces,dim=1);cache_error=errors(cached,standard)
        assert cache_error['relative_l2']<=1e-4,cache_error
        records.append({'case':case,'tokens':len(ids),'custom_vs_transformers':result,
                        'standard_cache_vs_full':cache_error,'argmax_matches':matches,'status':'PASS'})
report={'source':'original Q8 GGUF dequantized to FP32; no official BF16 quality claim',
        'scope':'all 24 language layers and all 151674 logits, plus independent standard KV-cache path',
        'cases':records,'status':'PASS'}
(a.source/'standard-export-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
