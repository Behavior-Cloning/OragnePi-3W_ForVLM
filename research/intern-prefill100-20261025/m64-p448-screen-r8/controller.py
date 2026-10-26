from pathlib import Path
import json,sys,hashlib,shlex,os,time,tarfile,traceback
R=Path(__file__).resolve().parent;T=R.parent;sys.path.insert(0,str(T));from controller_down_outputfp16_tmpfs import b,locked
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();remote='/tmp/opi-m64-p448-screen-r8';state=R/'controller.json';assert not state.exists()
s={'status':'preparing','pid':os.getpid(),'started':time.time(),'remote':remote}
def save():state.write_text(json.dumps(s,indent=2)+'\n')
save()
try:
 binary=R/'probe';assert sha(binary)==json.loads((R/'probe-build.json').read_text())['binary_sha256']
 files={'probe':binary,'board_probe_compact.py':R/'board_probe.py'};cfg={'run':'m64-p448-screen-r8','warmup':3,'repeats':30,'binary':remote+'/probe','binary_sha256':sha(binary),'sequence':[],'scope':'L0/L1 and L2/L3; M64P128 vsP448;M32P448 throughput reference;effectivehistory128;CPU7;no full TTFT'}
 for layer in [0,2]:
  parent64=T/'compact-m64-r4/graphs'/('l'+str(layer));parent32=T/'down-outlier-split-r2/full-c448-m32' if layer==2 else T/'capacity448-m32/cases/prefill32/l0'
  for name,rows,past,g in [(f'l{layer}-m32-before',32,448,parent32),(f'l{layer}-m64-before',64,128,parent64),(f'l{layer}-m64-p448',64,448,R/('l'+str(layer))),(f'l{layer}-m64-after',64,128,parent64),(f'l{layer}-m32-after',32,448,parent32)]:
   nb=g/'export_nbg_unify/network_binary.nb';inp=R/'fixtures'/f'l{layer}-m{rows}-p{past}'/'input.f32';nbname=f'l{layer}-m{rows}-p{past}.nb';inname=f'l{layer}-m{rows}-p{past}.f32';files[nbname]=nb;files[inname]=inp
   cfg['sequence'].append({'run_id':name,'query_tokens':rows,'first_layer':layer,'capacity':past,'nb':remote+'/'+nbname,'nb_sha256':sha(nb),'input':remote+'/'+inname,'input_sha256':sha(inp)})
 (R/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');files['config.json']=R/'config.json';expected={n:sha(p) for n,p in files.items()}
 with b.hostlock():
  locked('python3 -c '+shlex.quote("import shutil;from pathlib import Path;m=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024;assert m>1200*1024**2 and shutil.disk_usage('/tmp').free>1000*1024**2"),check=True)
  locked('mkdir '+remote,check=True)
  for n,p in files.items():b.scp(str(p),b.HOST+':'+remote+'/'+n,check=True,timeout=120)
  locked('sha256sum -c -',input=''.join(h+'  '+remote+'/'+n+'\n' for n,h in expected.items()).encode(),capture_output=True,check=True);locked('chmod +x '+remote+'/probe',check=True)
  (R/'deployment-sha.json').write_text(json.dumps(expected,indent=2)+'\n')
  cmd='flock -w 5 /run/opi-npu-kv-benchmark.lock timeout -k 10 400 python3 '+remote+'/board_probe_compact.py --config '+remote+'/config.json --out '+remote+'/results';wrapper=cmd+' >'+remote+'/process.log 2>&1; rc=$?; echo "$rc" >'+remote+'/exit'
  z=b.ssh('nohup sh -c '+shlex.quote(wrapper)+' </dev/null >/dev/null 2>&1 & echo $!',capture_output=True,text=True,check=True,timeout=20);s.update(status='running',board_pid=int(z.stdout.strip()));save()
  for _ in range(50):
   time.sleep(5);z=b.ssh('cat '+remote+'/exit',capture_output=True,text=True,timeout=20)
   if z.returncode==0:s['returncode']=int(z.stdout.strip());break
  else:raise TimeoutError('observation timeout; do not restart')
  z=locked('tar -czf - -C '+remote+' results process.log exit',capture_output=True,check=True,timeout=60);(R/'results.tar.gz').write_bytes(z.stdout)
  with tarfile.open(R/'results.tar.gz') as tar:
   for m in tar.getmembers():assert not m.issym() and not m.islnk() and '..' not in Path(m.name).parts
   tar.extractall(R)
  meta=json.loads((R/'results/meta.json').read_text());s.update(status='complete' if s['returncode']==0 and meta['status']=='complete' else 'failed',ended=time.time());save()
except BaseException:s.update(status='failed',error=traceback.format_exc(),ended=time.time());save();raise
