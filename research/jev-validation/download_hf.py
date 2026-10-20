import subprocess,json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parent
repo='chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF';m=json.loads((root/'sources/hf-Jev-Style-0.8B-Decision-v3-GGUF.json').read_text());rev=m['sha'];d=root/'models/jev08';d.mkdir(exist_ok=True)
files=[x['rfilename'] for x in m['siblings'] if not x['rfilename'].startswith(('figures/','.')) and (not x['rfilename'].endswith('.gguf') or x['rfilename'].endswith('Q4_K_M.gguf'))];manifest={'repo':repo,'revision':rev,'source':'hf-mirror.com','fallback_reason':'ModelScope same repository returned 404','files':[],'status':'running'}
try:
 for fn in files:
  p=d/fn;p.parent.mkdir(exist_ok=True,parents=True)
  if not p.exists():
   subprocess.run(['curl','-fsSL','--retry','3','--connect-timeout','15','--max-time','900','-C','-','-o',str(p)+'.part',f'https://hf-mirror.com/{repo}/resolve/{rev}/{fn}'],check=True);Path(str(p)+'.part').replace(p)
  h=hashlib.sha256(p.read_bytes()).hexdigest();manifest['files'].append({'file':fn,'bytes':p.stat().st_size,'sha256':h});print(fn,'OK',flush=True)
 manifest['status']='complete'
except Exception as e:manifest['status']='failed';manifest['error']=repr(e)
(d/'download-manifest.json').write_text(json.dumps(manifest,indent=2))
