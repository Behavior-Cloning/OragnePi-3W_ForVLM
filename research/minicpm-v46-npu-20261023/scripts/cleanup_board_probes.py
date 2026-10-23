from pathlib import Path
import sys,json,fcntl,shlex,subprocess
r=Path(__file__).resolve().parents[1];sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
lock=Path('/opt/opi-lab/restored-slim3-20261017/projects/orangepi-zero3w/npu/results/board-host.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
d='/opt/opi-vlm/minicpm-v46-npu-20261023'
code="""from pathlib import Path
import json,hashlib,shutil,time
r=Path('/opt/opi-vlm/minicpm-v46-npu-20261023');before=shutil.disk_usage('/').free;rows=[]
for p in r.glob('*.nb'):
 assert p.is_file() and not p.is_symlink();rows.append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()});p.unlink()
j={'time':time.time(),'removed_probes':rows,'bytes':sum(v['bytes'] for v in rows),'free_before':before,'free_after':shutil.disk_usage('/').free,'preserved_model':str(r/'baseline-fp16'),'preserved_model_variant':'epsilon-safe FP16 text prototype, not production qualified'};(r/'probe-cleanup.json').write_text(json.dumps(j,indent=2));print(json.dumps(j,indent=2))
"""
cmd='python3 -c '+shlex.quote(code);z=b.ssh('flock -n /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),capture_output=True,text=True,timeout=30);assert z.returncode==0,z.stderr
(r/'results/board-probe-cleanup.json').write_text(z.stdout)
z=b.ssh('flock -n /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote('df -h /; free -m; ps -p 967,976 -o pid,comm,etime,rss; cat /sys/kernel/debug/viplite/mem_profile'),capture_output=True,text=True,timeout=20);(r/'results/board-final-state.txt').write_text(z.stdout+z.stderr);print(z.stdout,flush=True)
