from pathlib import Path
import json
from tokenizers import Tokenizer
R=Path(__file__).resolve().parent;f=json.loads((R.parent/'results/ttft-32-64-matrix/fixtures.json').read_text());tok=Tokenizer.from_file('/opt/opi-lab/source-validation-20261017/official-InternVL3-1B/tokenizer.json');enc=lambda s:tok.encode(s,add_special_tokens=False).ids
prefix=f['prefix'];header='<|im_start|>user\n';result=dict(system_total=len(enc(prefix)),system_plus_user_header=len(enc(prefix+header)),fixed_user_header_extra=len(enc(prefix+header))-len(enc(prefix)),system_batch_reused=len(enc(prefix))//16*16,plus_header_batch_reused=len(enc(prefix+header))//16*16,body_to_new={str(n):n+8+len(enc(prefix))%16 for n in [32,48,64]},conclusion='User header caching alone does not cross next16 boundary; no batch reduction without exact/changed partition.')
(R/'fixed-template-token-counts.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
