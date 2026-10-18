import json,shlex,sys,hashlib
from pathlib import Path
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
repo=Path(__file__).resolve().parents[2];remote='/opt/opi-vlm/asr-small-20261018';out=repo/'results/asr-small-20261018'
with b.hostlock():
 cmd=f'cd {remote} && PYTHONPATH=/opt/opi-vlm/audio-lab-20261018/deps OPENBLAS_NUM_THREADS=1 taskset -c 6,7 python3 worker.py --kind qwen --indices 0,8,16 --repeats 1 --out qwen-bigcore.json'
 with (out/'qwen-bigcore.log').open('w') as log:r=b.ssh('flock -w 30 /run/opi-npu-kv-benchmark.lock bash -o pipefail -c '+shlex.quote(cmd),stdout=log,stderr=log,stdin=__import__('subprocess').DEVNULL,timeout=600)
 r.check_returncode();result=b.ssh('flock -w 30 /run/opi-npu-kv-benchmark.lock cat '+remote+'/qwen-bigcore.json',capture_output=True,check=True,timeout=30);(out/'qwen-bigcore.json').write_bytes(result.stdout)
print((out/'qwen-bigcore.log').read_text())
