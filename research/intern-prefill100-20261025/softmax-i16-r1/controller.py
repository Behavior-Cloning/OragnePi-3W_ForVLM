from pathlib import Path
import json,sys,hashlib,shlex,os,time,tarfile,traceback
R=Path(__file__).resolve().parent;T=R.parent;sys.path.insert(0,str(T));from controller_down_outputfp16_tmpfs import b,locked
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();remote='/tmp/opi-softmax-i16-r1';state=R/'controller.json';assert not state.exists()
s={'status':'preparing','pid':os.getpid(),'started':time.time(),'remote':remote}
def save():state.write_text(json.dumps(s,indent=2)+'\n')
save()
try:
 source=T/'compact-m64-r4/graphs/l0';candidate=R/'m64-p128';meta=json.loads((candidate/'build.json').read_text());assert meta['payload_equal'] and len(meta['quantizer_changes'])==2
 binary=T/'runtime/compact-probe/tensor-probe-compact';assert sha(binary)=='18b29578182bf1067512008ad914897a7639bcdba6d89dff2bc4bee173f4c1e4'
 inp=T/'fixtures-compact/c128-m64-l0/probe-input.f32';fm=json.loads((inp.parent/'probe-fixture.json').read_text());assert sha(inp)==fm['input_sha256'] and fm['history']==128 and fm['query_tokens']==64 and fm['same_all_active_KV']
 files={'probe':binary,'board_probe_compact.py':T/'board_probe_compact.py','input.f32':inp,'parent.nb':source/'export_nbg_unify/network_binary.nb','i16.nb':candidate/'export_nbg_unify/network_binary.nb'}
 cfg={'run':'softmax-i16-r1-m64','warmup':3,'repeats':30,'binary':remote+'/probe','binary_sha256':sha(binary),'sequence':[],'scope':'two layers L0/L1 only; original M64 past128 vsI16 softmax input; canonical input, not TTFT'}
 for name,nb in [('before','parent.nb'),('i16','i16.nb'),('after','parent.nb')]:cfg['sequence'].append({'run_id':name,'query_tokens':64,'first_layer':0,'capacity':128,'nb':remote+'/'+nb,'nb_sha256':sha(files[nb]),'input':remote+'/input.f32','input_sha256':sha(inp)})
 (R/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');files['config.json']=R/'config.json';expected={n:sha(p) for n,p in files.items()}
 with b.hostlock():
  locked('python3 -c '+shlex.quote("import shutil;from pathlib import Path;m=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024;assert m>1400*1024**2 and shutil.disk_usage('/tmp').free>1200*1024**2"),check=True)
  locked('mkdir '+remote,check=True)
  for n,p in files.items():b.scp(str(p),b.HOST+':'+remote+'/'+n,check=True,timeout=120)
  locked('sha256sum -c -',input=''.join(h+'  '+remote+'/'+n+'\n' for n,h in expected.items()).encode(),capture_output=True,check=True);locked('chmod +x '+remote+'/probe',check=True)
  (R/'deployment-sha.json').write_text(json.dumps(expected,indent=2)+'\n')
  cmd='flock -w 5 /run/opi-npu-kv-benchmark.lock timeout -k 10 300 python3 '+remote+'/board_probe_compact.py --config '+remote+'/config.json --out '+remote+'/results'
  wrapper=cmd+' >'+remote+'/process.log 2>&1; rc=$?; echo "$rc" >'+remote+'/exit'
  z=b.ssh('nohup sh -c '+shlex.quote(wrapper)+' </dev/null >/dev/null 2>&1 & echo $!',capture_output=True,text=True,check=True,timeout=20);s.update(status='running',board_pid=int(z.stdout.strip()));save()
  for _ in range(40):
   time.sleep(5);z=b.ssh('cat '+remote+'/exit',capture_output=True,text=True,timeout=20)
   if z.returncode==0:s['returncode']=int(z.stdout.strip());break
  else:raise TimeoutError('observation timeout; do not restart')
  z=locked('tar -czf - -C '+remote+' results process.log exit',capture_output=True,check=True,timeout=60);(R/'results.tar.gz').write_bytes(z.stdout)
  with tarfile.open(R/'results.tar.gz') as tar:
   for m in tar.getmembers():assert not m.issym() and not m.islnk() and '..' not in Path(m.name).parts
   tar.extractall(R)
  meta=json.loads((R/'results/meta.json').read_text());s.update(status='complete' if s['returncode']==0 and meta['status']=='complete' else 'failed',ended=time.time());save()
except BaseException:s.update(status='failed',error=traceback.format_exc(),ended=time.time());save();raise
