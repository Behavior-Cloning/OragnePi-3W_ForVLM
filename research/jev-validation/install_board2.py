from pathlib import Path
import sys
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
with b.hostlock():
 r=b.ssh("flock -w 10 /run/opi-npu-kv-benchmark.lock sh -c 'apt-get update -qq && apt-get install -y -qq python3.10-venv && mkdir -p /opt/opi-vlm/jev-lab-20261020 && python3 -m venv /opt/opi-vlm/jev-lab-20261020/venv; /opt/opi-vlm/jev-lab-20261020/venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==2.7.0 transformers==4.57.1 laya==0.4.12 psutil' > /opt/opi-vlm/jev-install-20261020-attempt2.log 2>&1",timeout=1200)
 print('install_exit',r.returncode,flush=True)
 b.scp(b.HOST+':/opt/opi-vlm/jev-install-20261020-attempt2.log',Path(__file__).parent/'results/board-install-attempt2.log',timeout=60)
