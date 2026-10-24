from pathlib import Path
import json,hashlib
import numpy as np
p=Path(__file__).resolve().parent/'results/ttft-aligned-matrix-v2';rows=[json.loads(x) for x in (p/'events.jsonl').read_text().splitlines()]
d={v:{x['case']:x for x in rows if x['event']=='diagnostic' and x['variant']==v} for v in ['plain','tail8','group','mask','group-repeat','mask-validate']}
out={'expected_bytes':4*151674*4,'cases':{},'all_files_checked':0}
for case in d['plain']:
 arrays={}
 for v in d:
  x=d[v][case];raw=(p/x['logits_file']).read_bytes();assert len(raw)==out['expected_bytes'] and hashlib.sha256(raw).hexdigest()==x['sha256']
  arrays[v]=np.frombuffer(raw,dtype='<f4').reshape(4,151674).astype(np.float64);assert np.isfinite(arrays[v]).all();out['all_files_checked']+=1
 a,b=arrays['plain'],arrays['mask'];delta=b-a
 out['cases'][case]={'plain_to_mask_relative_l2':float(np.linalg.norm(delta)/np.linalg.norm(a)),'plain_to_mask_max_abs':float(np.abs(delta).max()),'plain_to_mask_argmax_equal_steps':int((a.argmax(1)==b.argmax(1)).sum()),'steps':4,'group_to_mask_bitwise_equal':d['group'][case]['sha256']==d['mask'][case]['sha256'],'all_optimized_variants_bitwise_equal':len({d[v][case]['sha256'] for v in d if v!='plain'})==1}
(p/'numerical-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
