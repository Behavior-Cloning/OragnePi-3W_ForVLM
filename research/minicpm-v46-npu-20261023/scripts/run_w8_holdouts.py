from pathlib import Path
import sys,subprocess,shlex
r=Path(__file__).resolve().parents[1];bld=Path('/opt/opi-lab/minicpm-v46-npu-20261023');sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
with b.hostlock():
 d='/opt/opi-vlm/minicpm-v46-npu-20261023'
 def run(cmd,timeout=40):
  z=b.ssh('flock -n /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),capture_output=True,text=True,timeout=timeout);print(z.stdout,z.stderr,flush=True);assert z.returncode==0;return z
 run('mkdir -p '+d+'/holdout-prompts')
 b.scp(r/'scripts/mini_runtime_cases.c',b.HOST+':'+d+'/mini_runtime_cases.c',check=True,timeout=30)
 for i in range(4):b.scp(bld/'holdout-prompts'/(str(i)+'.i32'),b.HOST+':'+d+'/holdout-prompts/'+str(i)+'.i32',check=True,timeout=30)
 run('cd '+d+' && cc -O2 -Wall -Wextra -Wno-misleading-indentation mini_runtime_cases.c -lNBGlinker -lVIPhal -lm -o mini_runtime_cases')
 z=run('cd '+d+' && timeout --kill-after=10s 180s env MINI_PROMPTS_DIR='+d+'/holdout-prompts taskset -c 6,7 ./mini_runtime_cases '+d+'/w8scaled-all '+d+'/holdout-prompts/0.i32 '+d+'/w8scaled-holdouts.jsonl 4',195)
 (r/'results/w8scaled-holdouts.log').write_text(z.stdout+z.stderr);b.scp(b.HOST+':'+d+'/w8scaled-holdouts.jsonl',r/'results/w8scaled-holdouts.jsonl',check=True,timeout=30)
print('COMPLETE',flush=True)
