from pathlib import Path
import subprocess,json,hashlib,time
root=Path(__file__).resolve().parent
MODELS=[
('laya','convaiinnovations/laya-multilingual',['README.md','config.json','encoder/config.json','model.safetensors','rl_agent_config.json','tokenizer/tokenizer.json','tokenizer/tokenizer_config.json']),
('jev08','chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF',None),
('macjev','chaoliangUNSW/MacJev-322M-4K-Laya',['README.md','encoder/config.json','model.safetensors','rl_agent_config.json','tokenizer/tokenizer.json','tokenizer/tokenizer_config.json','LICENSE','NOTICE']),
]
def curl(url,dest):
 dest.parent.mkdir(parents=True,exist_ok=True)
 subprocess.run(['curl','-fL','--retry','3','--retry-delay','2','--connect-timeout','15','--max-time','900','-C','-','-o',str(dest)+'.part',url],check=True)
 Path(str(dest)+'.part').replace(dest)
for name,repo,allow in MODELS:
 api=root/'sources'/f'{name}-api.json'
 curl('https://hf-mirror.com/api/models/'+repo,api)
 meta=json.loads(api.read_text());rev=meta['sha']
 if allow is None:
  allow=[s['rfilename'] for s in meta['siblings'] if not s['rfilename'].startswith(('figures/','validation/')) and (s['rfilename'].endswith(('.py','.sh','.cpp','.h','.hpp','.json','.txt','.md')) or s['rfilename'].endswith('Q4_K_M.gguf'))]
 records=[]
 for fn in allow:
  target=root/'models'/name/fn
  if not target.exists(): curl(f'https://hf-mirror.com/{repo}/resolve/{rev}/{fn}',target)
  h=hashlib.sha256()
  with target.open('rb') as f:
   while chunk:=f.read(8*1024*1024):h.update(chunk)
  records.append(dict(file=fn,bytes=target.stat().st_size,sha256=h.hexdigest()))
 (root/'models'/name/'download-manifest.json').write_text(json.dumps(dict(repo=repo,revision=rev,source='https://hf-mirror.com',files=records),indent=2))
 print('COMPLETE',name,rev,sum(r['bytes'] for r in records),flush=True)

