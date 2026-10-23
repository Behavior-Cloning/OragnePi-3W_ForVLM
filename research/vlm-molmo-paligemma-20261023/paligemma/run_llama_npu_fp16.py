import pathlib,sys,subprocess,time,tarfile,shlex
P=pathlib.Path(__file__).parent;sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
B='/opt/opi-vlm/llama32-compat-20261023';R=P.parents[1]/'llama32-compat-20261023/board-results-mlp-fp16';R.mkdir(exist_ok=True)
with b.hostlock():
 b.scp('/opt/opi-lab/llama32-compat-20261023/mlp-fp16/export_nbg_unify/network_binary.nb',b.HOST+':'+B+'/network_mlp_fp16.nb',timeout=180).check_returncode()
 cmd='cd '+B+' && mkdir -p outputs-mlp-fp16 && timeout --kill-after=5s 60s ./layer_probe network_mlp_fp16.nb fixtures outputs-mlp-fp16 >probe-mlp-fp16.log 2>&1'
 r=b.ssh('flock -n -E 73 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),timeout=80);(R/'runner-exit.txt').write_text(str(r.returncode));print('MLP FP16 probe exit',r.returncode,flush=True)
 b.ssh('flock -n -E 73 /run/opi-npu-kv-benchmark.lock tar -cf '+B+'/results-mlp-fp16.tar -C '+B+' probe-mlp-fp16.log outputs-mlp-fp16',timeout=20)
 b.scp(b.HOST+':'+B+'/results-mlp-fp16.tar',str(R/'results.tar'),timeout=30)
 with tarfile.open(R/'results.tar') as t:t.extractall(R,filter='data')
 (R/'probe-mlp-fp16.log').rename(R/'probe.log');(R/'outputs-mlp-fp16').rename(R/'outputs')
