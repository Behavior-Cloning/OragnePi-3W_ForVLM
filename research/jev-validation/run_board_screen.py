from pathlib import Path
import sys,subprocess,json
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
r=Path(__file__).resolve().parent;dest='/opt/opi-vlm/jev-lab-20261020'
with b.hostlock():
 for name in ['laya','nano33']:
  subprocess.run(['sshpass','-e','rsync','-rt','--partial','--append-verify','-e','ssh '+' '.join(b.COMMON),str(r/'models'/name),b.HOST+':'+dest+'/models/'],env=b.env(),check=True,timeout=900)
 for name in ['smoke.py','board_adapters.py','board_bench.py','cases.json','actions.json','groups.json']:
  b.scp(r/name,b.HOST+':'+dest+'/'+name,check=True,timeout=40)
 check="import pathlib,json,hashlib; r=pathlib.Path('"+dest+"/models'); "
 check+="[(print(m['repo'],f['file'],'SHA_OK') if hashlib.sha256((p.parent/f['file']).read_bytes()).hexdigest()==f['sha256'] else (_ for _ in ()).throw(ValueError(f['file']))) for p in r.glob('*/download-manifest.json') for m in [json.loads(p.read_text())] for f in m['files']]"
 import shlex
 b.ssh('python3 -c '+shlex.quote(check),check=True,timeout=90)
 b.ssh("flock -w 10 /run/opi-npu-kv-benchmark.lock sh -c 'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false; taskset -c 6,7 "+dest+"/venv/bin/python "+dest+"/smoke.py' > "+dest+"/results/smoke.log 2>&1",timeout=180)
 b.scp(b.HOST+':'+dest+'/results/smoke.log',r/'results/smoke.log',check=True)
 for model in ['nano33','laya']:
  ret=b.ssh("flock -w 10 /run/opi-npu-kv-benchmark.lock sh -c 'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false; taskset -c 6,7 "+dest+"/venv/bin/python -u "+dest+"/board_bench.py --model "+model+"' > "+dest+"/results/"+model+"-bench.log 2>&1",timeout=3600)
  print(model,'exit',ret.returncode,flush=True)
  subprocess.run(['sshpass','-e','rsync','-rt','-e','ssh '+' '.join(b.COMMON),b.HOST+':'+dest+'/results/',str(r/'results/board')+'/'],env=b.env(),check=True,timeout=120)
