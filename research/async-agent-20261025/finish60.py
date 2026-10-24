from pathlib import Path
import json,time,subprocess,sys
p=Path(__file__).resolve().parent
while True:
 s=json.loads((p/'stress60-v2-controller.json').read_text())
 if s['status'] in ['finished','failed']:break
 time.sleep(5)
if s.get('returncode')!=0:raise RuntimeError(s)
subprocess.run(['python3',str(p/'summarize.py'),str(p/'results/stress60-v2')],stdout=(p/'stress60-summary-stdout.json').open('w'),check=True)
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
with b.hostlock():
 r=b.ssh("flock -w 5 /run/opi-npu-kv-benchmark.lock sh -c 'uptime; free -m; df -h /; cat /sys/kernel/debug/viplite/mem_profile; ps -eo pid,comm,rss,pcpu --sort=-rss | head -15; dmesg | tail -80'",capture_output=True,text=True,timeout=25)
 (p/'results/stress60-v2/post-health.txt').write_text(r.stdout+'\n'+r.stderr)
(p/'stress60-finish-status.json').write_text(json.dumps({'status':'complete','health_exit':r.returncode}))
