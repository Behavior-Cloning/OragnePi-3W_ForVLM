from pathlib import Path
import sys,subprocess,json,time,shlex
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
r=Path(__file__).resolve().parent;d='/opt/opi-vlm/jev-lab-20261020'
with b.hostlock():
 b.ssh('mkdir -p '+d+'/bin; dmesg > '+d+'/results/dmesg-before-pairs.txt',check=True)
 for src,dst in [(r/'models/jev08',d+'/models/'),(r/'build-arm64/bin/',d+'/bin/'),(r/'bin/',d+'/bin/')]:
  subprocess.run(['sshpass','-e','rsync','-rtL','--partial','--append-verify','-e','ssh '+' '.join(b.COMMON),str(src)+('/' if dst.endswith('/bin/') else ''),b.HOST+':'+dst],env=b.env(),check=True,timeout=1200)
 for name in ['board_adapters.py','board_bench.py','board_co_resident.py','resident-inputs.json','jev_style_decision_board.py']:
  b.scp(r/name,b.HOST+':'+d+'/'+name,check=True,timeout=40)
 b.scp(r.parents[1]/'npu/kv-cache/resident_worker.py',b.HOST+':'+d+'/resident_worker.py',check=True)
 b.ssh('chmod +x '+d+'/bin/jev-score '+d+'/bin/resident-smol; file '+d+'/bin/jev-score; ldd '+d+'/bin/jev-score',check=True)
 for script,label in [('board_bench.py --model laya --repeats 30 --warmup 5 --max-cases 20','laya-screen'),('board_bench.py --model jev08 --repeats 20 --warmup 3 --max-cases 12','jev08'),('board_co_resident.py','co-resident')]:
  cmd='export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false; timeout --kill-after=15s 3500s taskset -c 6,7 '+d+'/venv/bin/python -u '+d+'/'+script
  result=b.ssh('flock -w 10 /run/opi-npu-kv-benchmark.lock bash -o pipefail -c '+shlex.quote(cmd+' 2>&1 | tee '+d+'/results/'+label+'-run.log'),timeout=3600)
  print(label,'exit',result.returncode,flush=True)
  subprocess.run(['sshpass','-e','rsync','-rt','-e','ssh '+' '.join(b.COMMON),b.HOST+':'+d+'/results/',str(r/'results/board')+'/'],env=b.env(),check=True,timeout=180)
 b.ssh('dmesg > '+d+'/results/dmesg-after-pairs.txt; free -m; df -h /',check=True)
