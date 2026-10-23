import pathlib,sys,subprocess,time,tarfile
P=pathlib.Path(__file__).parent;sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
while pathlib.Path('/proc/6727').exists():time.sleep(2)
B='/opt/opi-vlm/llama32-compat-20261023';R=P.parents[1]/'llama32-compat-20261023/board-results';R.mkdir(exist_ok=True)
with b.hostlock():
 b.ssh('flock -n -E 73 /run/opi-npu-kv-benchmark.lock mkdir -p '+B,timeout=20).check_returncode()
 b.scp(str(P/'llama-npu-probe.tar'),b.HOST+':'+B+'/probe.tar',timeout=120).check_returncode()
 cmd="cd "+B+" && tar -xf probe.tar && mkdir -p outputs && cc -O2 -DMODEL_DIM=2048 -DMODEL_KV=8 -DMODEL_HD=64 layer_probe.c -lNBGlinker -lVIPhal -lm -o layer_probe && timeout --kill-after=5s 60s ./layer_probe network_binary.nb fixtures outputs >probe.log 2>&1"
 import shlex
 r=b.ssh('flock -n -E 73 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),timeout=90)
 (R/'runner-exit.txt').write_text(str(r.returncode));print('NPU probe exit',r.returncode,flush=True)
 b.ssh('tar -cf '+B+'/results.tar -C '+B+' probe.log outputs',timeout=20)
 b.scp(b.HOST+':'+B+'/results.tar',str(R/'results.tar'),timeout=30)
 with tarfile.open(R/'results.tar') as t:t.extractall(R,filter='data')
