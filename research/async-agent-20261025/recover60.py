from pathlib import Path
import sys,json,subprocess,tarfile
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
p=Path(__file__).resolve().parent;r='/opt/opi-vlm/async-agent-20261025';out=p/'results/stress60-v2';out.mkdir(parents=True,exist_ok=True)
with b.hostlock():
 q=b.ssh('flock -w 5 /run/opi-npu-kv-benchmark.lock tar -czf '+r+'/stress60-recovery.tar.gz -C '+r+'/stress60-v2 meta.json events.jsonl resources.jsonl npu.log',capture_output=True,text=True,timeout=60);q.check_returncode()
 b.scp(b.HOST+':'+r+'/stress60-recovery.tar.gz',str(p/'stress60-recovery.tar.gz'),check=True,timeout=180)
 with tarfile.open(p/'stress60-recovery.tar.gz') as t:t.extractall(out)
 h=b.ssh('flock -w 5 /run/opi-npu-kv-benchmark.lock sh -c "uptime; free -m; df -h /; cat /sys/kernel/debug/viplite/mem_profile; ps -eo pid,comm,rss,pcpu --sort=-rss | head -15; dmesg | tail -80"',capture_output=True,text=True,timeout=20)
 (out/'post-health.txt').write_text(h.stdout+'\n'+h.stderr)
subprocess.run(['python3',str(p/'summarize.py'),str(out)],stdout=(p/'stress60-summary-stdout.json').open('w'),check=True)
(p/'stress60-recovery-status.json').write_text(json.dumps({'status':'complete','controller_outcome':'SSH communication timeout3840s; original failed evidence retained','board_meta':json.loads((out/'meta.json').read_text()),'post_health_rc':h.returncode},ensure_ascii=False,indent=2))
