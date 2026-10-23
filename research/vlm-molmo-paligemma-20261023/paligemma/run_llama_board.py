import pathlib,sys,subprocess
P=pathlib.Path(__file__).parent;sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
B='/opt/opi-vlm/vlm-new-models-20261023'
with b.hostlock():
 r=b.ssh("flock -n -E 73 /run/opi-npu-kv-benchmark.lock mkdir -p "+B,timeout=20);r.check_returncode()
 for src in [P/'llama_board_runner.py',P/'llama-zh/FuseChat-Llama-3.2-1B-Instruct-Q4_K_M.gguf']:
  b.scp(str(src),b.HOST+':'+B+'/'+src.name,timeout=600).check_returncode()
 r=b.ssh('flock -n -E 73 /run/opi-npu-kv-benchmark.lock timeout -k 15 420 python3 '+B+'/llama_board_runner.py',timeout=440)
 print('board_runner_exit',r.returncode,flush=True)
 for name in ['llama-board-results.jsonl','llama-board-telemetry.json','llama-server.log']:
  b.scp(b.HOST+':'+B+'/'+name,str(P/name),timeout=40)
