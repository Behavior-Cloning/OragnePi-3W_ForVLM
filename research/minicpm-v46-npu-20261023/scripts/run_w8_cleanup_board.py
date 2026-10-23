from pathlib import Path
import sys,hashlib,json,shlex,fcntl
r=Path(__file__).resolve().parents[1];bld=Path('/opt/opi-lab/minicpm-v46-npu-20261023');sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
h={n+'.nb':hashlib.sha256((bld/'compiled-epssafe'/n/'export_nbg_unify/network_binary.nb').read_bytes()).hexdigest() for n in ['l'+str(i) for i in range(24)]+['head'+str(i) for i in range(8)]};p=r/'results/baseline-retained-wsl-hashes.json';p.write_text(json.dumps(h,indent=2))
lock=Path('/opt/opi-lab/restored-slim3-20261017/projects/orangepi-zero3w/npu/results/board-host.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);d='/opt/opi-vlm/minicpm-v46-npu-20261023'
for p in [p,r/'scripts/cleanup_w8_board.py']:b.scp(p,b.HOST+':'+d+'/'+p.name,check=True,timeout=30)
z=b.ssh('flock -n /run/opi-npu-kv-benchmark.lock python3 '+d+'/cleanup_w8_board.py',capture_output=True,text=True,timeout=180);print(z.stdout,z.stderr,flush=True);assert z.returncode==0
b.scp(b.HOST+':'+d+'/w8-cleanup-board.json',r/'results/w8-cleanup-board.json',check=True,timeout=30)
z=b.ssh('flock -n /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote('df -h /; free -m; ps -eo pid,args | grep -E "mini_runtime|probe|resident" | head -15'),capture_output=True,text=True,timeout=20);(r/'results/w8-final-board-snapshot.txt').write_text(z.stdout+z.stderr);print(z.stdout,flush=True)
