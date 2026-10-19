"""Compare rewound KV with full recomputation on the same resident NPU graphs."""
import hashlib,json,struct,sys
from pathlib import Path
from resident_worker import ResidentWorker
root=Path(__file__).resolve().parent
out=root/'prefix-validation-attempt2';out.mkdir(exist_ok=False)
base=Path('/opt/opi-vlm/npu-kv-probe/wsl-fresh-peak155-20261018')
data=(base/'board-fresh-peak155-h1024-n32-20261018-prompt.i32').read_bytes()
tokens=list(struct.unpack('<'+'i'*(len(data)//4),data))
w=ResidentWorker(['taskset','-c','6,7',str(root/'resident'),str(base),'24','1280','151674',str(base/'embeddings.f32'),'16','151645'],root,out/'worker.log',dump_dir=out)
result={'ready':w.ready,'cases':[],'status':'running'}
def request(p,reset=False):
 return list(w.generate(p,max_new=8,reset=reset,diagnostic=True))[-1]
try:
 seed=tokens[:160]
 request(seed,True)
 cases=[('changed-after-64',tokens[:64]+tokens[200:264]),('changed-after-128',tokens[:128]+tokens[300:340]),('shorter',tokens[:80]),('repeat',tokens[:80]),('no-common',[1]+tokens[1:100])]
 for name,prompt in cases:
  if name=='changed-after-128':request(seed,True)
  reused=request(prompt)
  reset=request(prompt,True)
  a=(out/(str(reused['id'])+'-logits.f32')).read_bytes();b=(out/(str(reset['id'])+'-logits.f32')).read_bytes()
  row={'case':name,'reuse':reused,'reset':reset,'logits_bitwise_equal':a==b,'tokens_equal':reused['tokens']==reset['tokens'],'reuse_sha256':hashlib.sha256(a).hexdigest(),'reset_sha256':hashlib.sha256(b).hexdigest()}
  result['cases'].append(row);print(json.dumps(row),flush=True)
  (out/'result.json').write_text(json.dumps(result,indent=2))
 result['status']='passed' if all(x['logits_bitwise_equal'] for x in result['cases']) else 'failed'
finally:
 result['closed']=w.close();(out/'result.json').write_text(json.dumps(result,indent=2))
print(result['status'],flush=True)
if result['status']!='passed':sys.exit(1)
