from pathlib import Path
import sys,subprocess,time,json,os
f=Path(sys.argv[1]).open('x',buffering=1);p=subprocess.Popen(sys.argv[2:])
while True:
 x={'at':time.monotonic(),'temperatures_mC':{a.parent.name:int(a.read_text()) for a in Path('/sys/class/thermal').glob('thermal_zone*/temp')},'meminfo':Path('/proc/meminfo').read_text(),'cpu_stat':Path('/proc/stat').read_text().splitlines()[0],'loadavg':os.getloadavg(),'frequency_khz':{a.parents[1].name:int(a.read_text()) for a in Path('/sys/devices/system/cpu').glob('cpu[0-9]*/cpufreq/scaling_cur_freq')}}
 for name in ['status','stat','smaps_rollup']:
  try:x[name]=Path('/proc/'+str(p.pid)+'/'+name).read_text()
  except (FileNotFoundError,ProcessLookupError):pass
 f.write(json.dumps(x)+'\n')
 if p.poll() is not None:break
 if max(x['temperatures_mC'].values())>=78000:p.terminate();raise SystemExit('thermal guard')
 time.sleep(0.5)
raise SystemExit(p.returncode)
