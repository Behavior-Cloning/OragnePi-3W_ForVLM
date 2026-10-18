import argparse,hashlib,io,json,shlex,sys,tarfile,subprocess
from pathlib import Path
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
p=argparse.ArgumentParser();p.add_argument('--dynamic',type=int,choices=[64,256],default=256);p.add_argument('--mode',required=True);p.add_argument('--name',required=True);p.add_argument('--deploy',action='store_true');p.add_argument('--deploy-qwen',action='store_true');p.add_argument('--kinds',default='stream,offline,para,sense');a=p.parse_args()
assert a.name.replace('-','').isalnum();repo=Path(__file__).resolve().parents[2];assets=Path('/mnt/f/Files/OPi-Lab/asr-small-models-20261018');out=repo/'results/asr-small-20261018';remote='/opt/opi-vlm/asr-small-20261018'
snap=out/(a.name+'-scripts');snap.mkdir(exist_ok=True)
for f in ['worker.py','coordinator.py','run_wsl.py']:(snap/f).write_bytes((repo/'research/asr-small'/f).read_bytes())
with b.hostlock():
 buf=io.BytesIO()
 with tarfile.open(fileobj=buf,mode='w') as t:
  for f in ['worker.py','coordinator.py']:t.add(repo/'research/asr-small'/f,arcname=f)
  if a.deploy_qwen:
   d=assets/'models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25';t.add(d,arcname='models/'+d.name)
  if a.deploy:
   t.add(assets/'models',arcname='models');t.add(assets/'corpus',arcname='corpus')
  manifest={f:hashlib.sha256((repo/'research/asr-small'/f).read_bytes()).hexdigest() for f in ['worker.py','coordinator.py']}
  for f in (assets/'models').rglob('*'):
   if f.is_file():manifest[str(f.relative_to(assets))]=hashlib.sha256(f.read_bytes()).hexdigest()
  for f in (assets/'corpus').rglob('*'):
   if f.is_file():manifest[str(f.relative_to(assets))]=hashlib.sha256(f.read_bytes()).hexdigest()
  raw=json.dumps(manifest,indent=2).encode();m=tarfile.TarInfo('manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
 (out/(a.name+'-manifest.json')).write_bytes(raw)
 b.ssh('flock -w 30 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(f'mkdir -p {remote} && tar -xf - -C {remote}'),input=buf.getvalue(),check=True,timeout=600)
 if a.deploy or a.deploy_qwen:
  verify="import hashlib,json,pathlib; r=pathlib.Path('.'); m=json.loads((r/'manifest.json').read_text()); assert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==h for p,h in m.items()); print('VERIFIED',len(m))"
  b.ssh('flock -w 30 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(f'cd {remote} && python3 -c '+shlex.quote(verify)),check=True,stdin=subprocess.DEVNULL,timeout=180)
 cmd=f'cd {remote} && PYTHONPATH=/opt/opi-vlm/audio-lab-20261018/deps OPENBLAS_NUM_THREADS=1 python3 -u coordinator.py --mode {shlex.quote(a.mode)} --out {a.name} --kinds {shlex.quote(a.kinds)} --dynamic {a.dynamic} 2>&1 | tee {a.name}.log'
 with (out/(a.name+'-live.log')).open('w') as live:
  result=b.ssh('flock -w 30 /run/opi-npu-kv-benchmark.lock bash -o pipefail -c '+shlex.quote(cmd),stdout=live,stderr=live,stdin=subprocess.DEVNULL,timeout=7200)
 packed=b.ssh('flock -w 30 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(f'cd {remote} && tar -czf - {a.name} {a.name}.log'),capture_output=True,check=True,stdin=subprocess.DEVNULL,timeout=180)
 dest=out/(a.name+'.tar.gz');dest.write_bytes(packed.stdout)
 with tarfile.open(dest) as t:
  for m in t.getmembers():assert not m.name.startswith('/') and '..' not in Path(m.name).parts and not m.issym() and not m.islnk()
  t.extractall(out)
 (out/(a.name+'-transfer.json')).write_text(json.dumps({'sha256':hashlib.sha256(packed.stdout).hexdigest(),'exit_code':result.returncode}))
 print((out/(a.name+'.log')).read_text());result.check_returncode()
