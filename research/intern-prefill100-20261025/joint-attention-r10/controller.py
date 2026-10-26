from pathlib import Path
import json,sys,hashlib,shlex,os,time,tarfile,traceback
R=Path(__file__).resolve().parent;T=R.parent;sys.path.insert(0,str(T));from controller_down_outputfp16_tmpfs import b,locked
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();remote='/tmp/opi-joint-attention-r10';state=R/'controller.json';assert not state.exists()
s={'status':'preparing','pid':os.getpid(),'started':time.time(),'remote':remote}
def save():state.write_text(json.dumps(s,indent=2)+'\n')
save()
try:
 binary=R/'probe';assert sha(binary)==json.loads((R/'probe-build.json').read_text())['binary_sha256']
 files={'probe':binary,'board_probe_compact.py':R/'board_probe.py'};cfg={'run':'joint-attention-r10','warmup':3,'repeats':30,'binary':remote+'/probe','binary_sha256':sha(binary),'sequence':[],'scope':'L2/L3 attention joint scores/value/both versus split;M32P448 andM64P128;history128;CPU7;nofullTTFT'}
 for name,rows,past,parent in [('m32-p448',32,448,T/'down-outlier-split-r2/full-c448-m32'),('m64-p128',64,128,T/'compact-m64-r4/graphs/l2')]:
  cases=[('before',parent)]+[(mode,R/(name+'-'+mode)) for mode in ['scores','value','both']]+[('after',parent)]
  for stage,g in cases:
   build=json.loads((R/(name+'-'+('scores' if stage in ['before','after'] else stage))/'build.json').read_text());assert build['status']=='compiled-unvalidated'
   nb=g/'export_nbg_unify/network_binary.nb';inp=T/'m64-p448-screen-r8/fixtures'/f'l2-m{rows}-p{past}'/'input.f32';nbname=name+'-'+('base' if stage in ['before','after'] else stage)+'.nb';inname=f'{name}.f32';files[nbname]=nb;files[inname]=inp
   assert sha(nb)==(build['source_sha256']['export_nbg_unify/network_binary.nb'] if stage in ['before','after'] else build['nb_sha256'])
   cfg['sequence'].append({'run_id':name+'-'+stage,'query_tokens':rows,'first_layer':2,'capacity':past,'nb':remote+'/'+nbname,'nb_sha256':sha(nb),'input':remote+'/'+inname,'input_sha256':sha(inp),'fixture_meta':json.loads((inp.parent/'meta.json').read_text())})
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
