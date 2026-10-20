import sys,time,json,subprocess
from pathlib import Path
sys.path.insert(0,"/mnt/f/Files/OPi-Lab/scripts");import board_access as b
r=Path(__file__).resolve().parent
with b.hostlock():
 b.ssh("mkdir -p /opt/opi-vlm/jev-lab-20261020/models /opt/opi-vlm/jev-lab-20261020/results")
 subprocess.run(["sshpass","-e","scp","-r",*b.COMMON,str(r/"models/laya"),b.HOST+":/opt/opi-vlm/jev-lab-20261020/models/"],env=b.env(),check=True,timeout=300)
 b.scp(r/"smoke.py",b.HOST+":/opt/opi-vlm/jev-lab-20261020/smoke.py",check=True)
 b.ssh("flock -w 10 /run/opi-npu-kv-benchmark.lock sh -c 'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; taskset -c 6,7 /opt/opi-vlm/jev-lab-20261020/venv/bin/python /opt/opi-vlm/jev-lab-20261020/smoke.py' > /opt/opi-vlm/jev-lab-20261020/results/smoke.log 2>&1",timeout=180)
 b.scp(b.HOST+":/opt/opi-vlm/jev-lab-20261020/results/smoke.log",r/"results/smoke.log")
