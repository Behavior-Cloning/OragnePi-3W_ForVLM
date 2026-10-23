from pathlib import Path
import sys,shlex,json,fcntl
r=Path(__file__).resolve().parents[1];sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
with b.hostlock():
 d='/opt/opi-vlm/minicpm-v46-npu-20261023';b.scp(r/'scripts/mini_runtime_grouped.c',b.HOST+':'+d+'/mini_runtime_grouped.c',check=True,timeout=30)
 def run(cmd,timeout=40):
  z=b.ssh('flock -n /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),capture_output=True,text=True,timeout=timeout);print(z.stdout,z.stderr,flush=True);assert z.returncode==0;return z
 run('cd '+d+' && cc -O2 -Wall -Wextra -Wno-misleading-indentation mini_runtime_grouped.c -lNBGlinker -lVIPhal -lm -o mini_runtime_grouped')
 z=run('cd '+d+' && timeout --kill-after=10s 180s taskset -c 6,7 ./mini_runtime_grouped '+d+'/w8scaled-all '+d+'/w8scaled-all/prompt.i32 '+d+'/w8grouped-runtime.jsonl 3',195)
 (r/'results/w8grouped-runtime.log').write_text(z.stdout+z.stderr);b.scp(b.HOST+':'+d+'/w8grouped-runtime.jsonl',r/'results/w8grouped-runtime.jsonl',check=True,timeout=30)
print('COMPLETE',flush=True)
