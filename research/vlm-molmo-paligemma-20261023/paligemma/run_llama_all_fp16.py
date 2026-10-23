import pathlib,sys,subprocess,time,tarfile,shlex,json,hashlib
P=pathlib.Path(__file__).parent;sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
B='/opt/opi-vlm/llama32-compat-20261023';R=P.parents[2]/'research/llama32-compat-20261023';D=R/'board-results-all-fp16';D.mkdir(exist_ok=True);src=pathlib.Path('/opt/opi-lab/llama32-compat-20261023/all-fp16/export_nbg_unify/network_binary.nb');expected='031202e00e8609a07b4434700ba15f1323a4831421239787b0fee7500bdce36f'
assert hashlib.sha256(src.read_bytes()).hexdigest()==expected
with b.hostlock():
 b.ssh('mkdir -p '+B+'/fixtures '+B+'/outputs-all-fp16',timeout=20).check_returncode()
 for f in [P/'llama-control-fixtures.tar',R/'layer_probe.c',R/'board_probe.c']:b.scp(str(f),b.HOST+':'+B+'/'+f.name,timeout=60).check_returncode()
 b.scp(str(src),b.HOST+':'+B+'/network_all_fp16.nb',timeout=180).check_returncode()
 r=b.ssh('sha256sum '+B+'/network_all_fp16.nb',timeout=30,capture_output=True,text=True);r.check_returncode();assert r.stdout.split()[0]==expected
 cmd='cd '+B+' && tar -xf llama-control-fixtures.tar -C fixtures && cc -O2 -DMODEL_DIM=2048 -DMODEL_KV=8 -DMODEL_HD=64 layer_probe.c -lNBGlinker -lVIPhal -lm -o layer_probe && timeout --kill-after=5s 60s ./layer_probe network_all_fp16.nb fixtures outputs-all-fp16 >probe-all-fp16.log 2>&1'
 r=b.ssh('flock -n -E 73 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),timeout=85);(D/'runner-exit.txt').write_text(str(r.returncode));print('all FP16 probe exit',r.returncode,flush=True)
 b.ssh('flock -n -E 73 /run/opi-npu-kv-benchmark.lock tar -cf '+B+'/results-all-fp16.tar -C '+B+' probe-all-fp16.log outputs-all-fp16',timeout=20).check_returncode()
 b.scp(b.HOST+':'+B+'/results-all-fp16.tar',str(D/'results.tar'),timeout=30).check_returncode()
 with tarfile.open(D/'results.tar') as t:t.extractall(D,members=[m for m in t.getmembers() if not m.issym() and not m.islnk()],filter='data')
 (D/'probe-all-fp16.log').rename(D/'probe.log');(D/'outputs-all-fp16').rename(D/'outputs')
 b.ssh('flock -n -E 73 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote('rm -- '+B+'/network_all_fp16.nb '+B+'/llama-control-fixtures.tar; rm -rf -- '+B+'/fixtures'),timeout=20).check_returncode()
 (P/'llama-all-fp16-cleanup.json').write_text(json.dumps(dict(path=B+'/network_all_fp16.nb',bytes=src.stat().st_size,sha256=expected,removed=True,fixture_staging_removed=True),indent=2));print('ALL_FP16_PROBE_COMPLETE_LOCK_RELEASE',flush=True)
