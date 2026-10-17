"""Run a command while recording WSL pressure and Docker cgroup memory peaks."""
import argparse
import datetime
import json
from pathlib import Path
import resource
import shutil
import subprocess
import time

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args();command=a.command[1:] if a.command and a.command[0]=='--' else a.command
if not command:raise ValueError('missing command')
a.output.mkdir(parents=True,exist_ok=False)
start=time.monotonic();peak_containers={};samples=0;min_available=None;max_swap=0
with (a.output/'stdout.log').open('w') as log,(a.output/'resources.jsonl').open('w') as resources:
    proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
    while True:
        mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split())>=3 and line.split()[2]=='kB'}
        containers={}
        for path in Path('/sys/fs/cgroup/system.slice').glob('docker-*.scope/memory.peak'):
            try:
                value=int(path.read_text());key=path.parent.name
                containers[key]=value;peak_containers[key]=max(value,peak_containers.get(key,0))
            except (FileNotFoundError,ValueError):pass
        row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start,
             'mem_total_bytes':mem['MemTotal'],'mem_available_bytes':mem['MemAvailable'],
             'swap_used_bytes':mem['SwapTotal']-mem['SwapFree'],'ext4_free_bytes':shutil.disk_usage(a.output).free,
             'docker_cgroup_peaks_bytes':containers}
        resources.write(json.dumps(row)+'\n');resources.flush();samples+=1
        min_available=min(row['mem_available_bytes'],min_available or row['mem_available_bytes']);max_swap=max(max_swap,row['swap_used_bytes'])
        if proc.poll() is not None:break
        time.sleep(1)
summary={'command':command,'exit_code':proc.returncode,'seconds':time.monotonic()-start,'samples':samples,
         'minimum_wsl_available_bytes':min_available,'maximum_wsl_swap_used_bytes':max_swap,
         'max_child_rss_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,
         'docker_cgroup_peaks_bytes':peak_containers,'status':'PASS' if proc.returncode==0 else 'FAIL'}
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));raise SystemExit(proc.returncode)
