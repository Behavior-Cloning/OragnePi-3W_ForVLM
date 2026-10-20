from pathlib import Path
import subprocess,sys,shlex,json
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
r=Path(__file__).resolve().parent;d='/opt/opi-vlm/jev-lab-20261020'
configs=json.loads((r/'results/selected-configs.json').read_text())
with b.hostlock():
 for n in ['board_co_resident.py','board_adapters.py','board_identity.py']:b.scp(r/n,b.HOST+':'+d+'/'+n,check=True)
 # Recompute board asset hashes under the same lock, outside all timed calls.
 b.ssh('flock -w 10 /run/opi-npu-kv-benchmark.lock '+d+'/venv/bin/python '+d+'/board_identity.py '+' '.join(configs)+' > '+d+'/results/final-pairs-board-identity.json',timeout=600,check=True)
 b.scp(b.HOST+':'+d+'/results/final-pairs-board-identity.json',r/'results/board/final-pairs-board-identity.json',check=True,timeout=60)
 for llm in ['smol500','intern1b','qwen15-cpu']:
  for model,cfg in configs.items():
   cool="import time;from pathlib import Path\nfor i in range(60):\n t=max(float(p.read_text())/1000 for p in Path('/sys/class/thermal').glob('thermal_zone*/temp'))\n if t<70:break\n time.sleep(5)\nelse:raise RuntimeError('did not cool below 70C')"
   b.ssh('flock -w 10 /run/opi-npu-kv-benchmark.lock '+d+'/venv/bin/python -c '+shlex.quote(cool),timeout=320,check=True)
   env={'CO_LABEL':'optimized-'+llm+'-'+model,'PAIR_MODELS':model,'PAIR_LLMS':llm,'PAIR_CONFIGS':json.dumps({model:cfg})}
   cmd='env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false '+' '.join(k+'='+shlex.quote(v) for k,v in env.items())+' timeout --kill-after=15s 600s taskset -c 6,7 '+d+'/venv/bin/python -u '+d+'/board_co_resident.py'
   z=b.ssh('flock -w 10 /run/opi-npu-kv-benchmark.lock bash -o pipefail -c '+shlex.quote(cmd+' 2>&1 | tee '+d+'/results/final-'+llm+'-'+model+'.log'),timeout=650)
   print(llm,model,'exit',z.returncode,flush=True)
   subprocess.run(['sshpass','-e','rsync','-rt','-e','ssh '+' '.join(b.COMMON),b.HOST+':'+d+'/results/',str(r/'results/board')+'/'],env=b.env(),check=True,timeout=180)
 b.ssh('dmesg > '+d+'/results/dmesg-final.txt; free -m; df -h /',check=True)
