from pathlib import Path
import sys,json,shlex,fcntl,numpy as np,hashlib,time
r=Path(__file__).resolve().parents[1];bld=Path('/opt/opi-lab/minicpm-v46-npu-20261023');sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
lock=Path('/opt/opi-lab/restored-slim3-20261017/projects/orangepi-zero3w/npu/results/board-host.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);d='/opt/opi-vlm/minicpm-v46-npu-20261023';rows=[]
for name in ['head0','l0','l3']:
 src=bld/'w8retry-source'/name
 for variant in sys.argv[1:] or ['compiled-epssafe','w8convert-perchannel']:
  kind=variant+'-'+name;nb=bld/variant/name/'export_nbg_unify/network_binary.nb'
  assert nb.exists()
  for p,n in [(src/'input.f32',kind+'-input.f32'),(nb,kind+'.nb')]:b.scp(p,b.HOST+':'+d+'/'+n,check=True,timeout=120)
  cmd='cd '+d+' && sha256sum '+kind+'.nb && env OPI_PROBE_WARMUP=5 timeout --kill-after=5s 60s taskset -c 6,7 ./probe '+kind+'.nb '+kind+'-input.f32 '+kind+'-output.f32 30 '+kind+'-timing.json'
  z=b.ssh('flock -n /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),capture_output=True,text=True,timeout=70);(r/'results'/(kind+'-board.log')).write_text(z.stdout+z.stderr);assert z.returncode==0,z.stderr;assert hashlib.sha256(nb.read_bytes()).hexdigest() in z.stdout
  for n in [kind+'-output.f32',kind+'-timing.json']:b.scp(b.HOST+':'+d+'/'+n,r/'results'/n,check=True,timeout=30)
  a=np.fromfile(r/'results'/(kind+'-output.f32'),np.float32);ref=np.fromfile(src/'reference.f32',np.float32);assert len(a)==len(ref);out=[];offset=0
  for n,shape in json.loads((src/'shapes.json').read_text())['outputs'].items():
   count=int(np.prod(shape));x=a[offset:offset+count];y=ref[offset:offset+count];offset+=count;out.append({'name':n,'max_abs':float(abs(x-y).max()),'relative_l2':float(np.linalg.norm(x-y)/np.linalg.norm(y)),'finite':bool(np.isfinite(x).all()),'cosine':float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)))})
  row={'name':name,'variant':variant,'timing':json.loads((r/'results'/(kind+'-timing.json')).read_text()),'errors':out,'nb_sha256':hashlib.sha256(nb.read_bytes()).hexdigest()};rows.append(row);(r/'results/w8retry-probes.json').write_text(json.dumps(rows,indent=2));print(row,flush=True)
print('COMPLETE',flush=True)
