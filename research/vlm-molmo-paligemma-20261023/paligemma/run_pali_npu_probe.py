import pathlib,sys,subprocess,shlex,tarfile,json,hashlib,time
P=pathlib.Path(__file__).parent;R=P.parents[2]/'research/paligemma2-npu-compat-20261023';W=pathlib.Path('/opt/opi-lab/paligemma2-npu-compat-20261023');B='/opt/opi-vlm/paligemma2-npu-compat-20261023';sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
with b.hostlock():
 code="import pathlib,shutil;assert shutil.disk_usage('/opt').free>1073741824+170000000;assert not any(p.read_text().strip()=='gemma-pali' for p in pathlib.Path('/proc').glob('[0-9]*/comm') if p.exists())"
 b.ssh('flock -n -E 73 /run/opi-npu-kv-benchmark.lock python3 -c '+shlex.quote(code),timeout=20).check_returncode()
 b.ssh('mkdir -p '+B+'/fixtures',timeout=20).check_returncode()
 for src in [R/'layer_probe.c',R/'board_probe.c',W/'board-fixtures.tar']:b.scp(str(src),b.HOST+':'+B+'/'+src.name,timeout=60).check_returncode()
 cmd='cd '+B+' && tar -xf board-fixtures.tar -C fixtures && cc -O2 -DMODEL_DIM=2304 -DMODEL_KV=4 -DMODEL_HD=256 layer_probe.c -lNBGlinker -lVIPhal -lm -o layer_probe'
 b.ssh('flock -n -E 73 /run/opi-npu-kv-benchmark.lock bash -c '+shlex.quote(cmd),timeout=60).check_returncode()
 cleanup=[]
 for variant in ['fp16','w8']:
  src=W/variant/'export_nbg_unify/network_binary.nb';sha=hashlib.sha256(src.read_bytes()).hexdigest();print('variant start',variant,time.time(),flush=True)
  b.scp(str(src),b.HOST+':'+B+'/network.nb',timeout=180).check_returncode()
  rr=b.ssh('sha256sum '+B+'/network.nb',timeout=30,capture_output=True,text=True);rr.check_returncode();assert rr.stdout.split()[0]==sha
  out=R/('board-results-'+variant);out.mkdir(exist_ok=True)
  cmd='cd '+B+' && mkdir -p outputs-'+variant+'; timeout -k 5 120 '+B+'/layer_probe '+B+'/network.nb '+B+'/fixtures '+B+'/outputs-'+variant+' >'+B+'/probe-'+variant+'.log 2>&1; rc=$?; echo "$rc" >'+B+'/exit-'+variant+'.txt; tar -cf '+B+'/result-'+variant+'.tar -C '+B+' probe-'+variant+'.log exit-'+variant+'.txt outputs-'+variant+'; exit "$rc"'
  rr=b.ssh('flock -n -E 73 /run/opi-npu-kv-benchmark.lock bash -c '+shlex.quote(cmd),timeout=145);print('variant exit',variant,rr.returncode,flush=True)
  b.scp(b.HOST+':'+B+'/result-'+variant+'.tar',str(out/'results.tar'),timeout=60).check_returncode()
  with tarfile.open(out/'results.tar') as t:t.extractall(out,members=[m for m in t.getmembers() if not m.issym() and not m.islnk()],filter='data')
  (out/'probe.log').write_text((out/('probe-'+variant+'.log')).read_text());(out/'runner-exit.txt').write_text(str(rr.returncode)+'\n');(out/'model-source.json').write_text(json.dumps(dict(path=str(src),bytes=src.stat().st_size,sha256=sha)))
  b.ssh('rm -- '+B+'/network.nb',timeout=20).check_returncode();cleanup.append(dict(path=B+'/network.nb',source=str(src),bytes=src.stat().st_size,sha256=sha,variant=variant,removed=True))
 # Only task fixture/tar staging, all originals and output files are preserved on F.
 cmd='rm -rf -- '+B+'/fixtures; rm -- '+B+'/board-fixtures.tar'
 b.ssh('flock -n -E 73 /run/opi-npu-kv-benchmark.lock bash -c '+shlex.quote(cmd),timeout=20).check_returncode()
 (P/'pali-npu-staging-cleanup.json').write_text(json.dumps(cleanup,indent=2));print('ALL_PALI_NPU_PROBES_COMPLETE_LOCK_RELEASE',flush=True)
