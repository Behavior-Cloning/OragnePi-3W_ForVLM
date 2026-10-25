import argparse,json,sys,time,traceback,hashlib,shlex
from pathlib import Path
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
p=Path(__file__).resolve().parent;a=argparse.ArgumentParser();a.add_argument('run');a.add_argument('--binary',required=True);a.add_argument('--cycles',type=int,default=1);args=a.parse_args();remote='/opt/opi-vlm/intern-nextbench-20261025/'+args.run
status=p/(args.run+'-controller.json');assert not status.exists();s={'status':'starting','started':time.time()}
def save():status.write_text(json.dumps(s,indent=2))
save()
try:
 cfg={'graph':'/opt/opi-vlm/intern-candidates-20261025/ctx256','env':{'OPI_TTFT_TAIL_MIN':'8','OPI_GROUP_PREFILL':'1','OPI_MASK_TEMPLATE':'1','OPI_GREEDY_I16_SINGLE_PASS':'1','OPI_PREFILL8_DIR':'/opt/opi-vlm/intern-candidates-20261025/tail8-c256'},'cycles':args.cycles,'binary_sha256':hashlib.sha256(Path(args.binary).read_bytes()).hexdigest()}
 cp=p/(args.run+'-config.json');cp.write_text(json.dumps(cfg,indent=2));dest=p/'results'/args.run;dest.mkdir(parents=True,exist_ok=False)
 with b.hostlock():
  b.ssh('mkdir '+remote,check=True,timeout=20)
  for name,src in {'board_close_smoke.py':p/'board_close_smoke.py','resident_worker.py':p/'resident_worker.py','aligned-inputs.json':p/'aligned-inputs.json','config.json':cp,'resident-test':Path(args.binary)}.items():b.scp(str(src),b.HOST+':'+remote+'/'+name,check=True,timeout=30)
  b.ssh('chmod 755 '+remote+'/resident-test',check=True,timeout=20);s['status']='running';save()
  z=b.ssh('flock -w 5 /run/opi-npu-kv-benchmark.lock timeout -k 10 240 python3 '+remote+'/board_close_smoke.py',capture_output=True,text=True,timeout=260)
  (dest/'process.log').write_text(z.stdout+z.stderr)
  for i in range(args.cycles):b.scp(b.HOST+':'+remote+'/stderr-'+str(i)+'.log',str(dest/('stderr-'+str(i)+'.log')),timeout=25)
  h=b.ssh('flock -w 5 /run/opi-npu-kv-benchmark.lock cat /sys/kernel/debug/viplite/mem_profile',capture_output=True,text=True,timeout=20);(dest/'post-memory.txt').write_text(h.stdout+h.stderr)
 s.update(status='finished' if z.returncode==0 else 'failed',returncode=z.returncode,ended=time.time());save()
except BaseException:s.update(status='failed',error=traceback.format_exc(),ended=time.time());save();raise
