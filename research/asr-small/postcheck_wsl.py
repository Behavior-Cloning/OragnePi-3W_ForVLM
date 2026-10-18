"""Read-only post-experiment health evidence under coordinator and board locks."""
import json,shlex,sys,hashlib
from pathlib import Path
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
repo=Path(__file__).resolve().parents[2]
script=r'''
import json,os,shutil,subprocess,time
from pathlib import Path
r=Path('/opt/opi-vlm/asr-small-20261018');mem={l.split(':')[0]:int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines()}
procs=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=p.read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if ('worker.py' in args and 'python' in args and int(p.parent.name)!=os.getpid() and ('/asr-small-20261018/' in args or '/asr-small-20261018' in str((p.parent/'cwd').resolve()))):procs.append({'pid':int(p.parent.name),'args':args})
 except OSError:pass
out={'page_size_bytes':os.sysconf('SC_PAGE_SIZE'),'cpuinfo':Path('/proc/cpuinfo').read_text(),'time':time.time(),'disk':shutil.disk_usage(r)._asdict(),'meminfo_kib':mem,'active_asr_workers':procs,'temperature_mC':Path('/sys/class/thermal/thermal_zone0/temp').read_text().strip(),'services':subprocess.run(['systemctl','is-active','opi-vlm-model.service','opi-vlm-web.service','opi-vlm-control.service'],capture_output=True,text=True).stdout,'vmstat':Path('/proc/vmstat').read_text(),'memory_pressure':Path('/proc/pressure/memory').read_text() if Path('/proc/pressure/memory').exists() else None}
print(json.dumps(out,indent=2))
'''
with b.hostlock():
 result=b.ssh('flock -w 30 /run/opi-npu-kv-benchmark.lock python3 -',input=script,text=True,capture_output=True,check=True,timeout=60)
 (repo/'results/asr-small-20261018/board-postcheck.json').write_text(result.stdout)
print(result.stdout)
