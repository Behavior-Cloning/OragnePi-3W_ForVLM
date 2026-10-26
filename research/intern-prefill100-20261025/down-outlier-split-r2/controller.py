from pathlib import Path
import os,sys,json,hashlib,shlex,tarfile,time,traceback
R=Path(__file__).resolve().parent;T=R.parent;sys.path.insert(0,str(T));from controller_down_outputfp16_tmpfs import b,locked,upload,extract
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
status=R/'controller.json';assert not status.exists(),'unique execution ID required'
remote='/tmp/opi-outlier8-20261026-r1';state={'status':'preparing','pid':os.getpid(),'started':time.time(),'remote':remote}
def save():status.write_text(json.dumps(state,indent=2))
save()
try:
 audit=json.loads((R/'STATIC_AUDIT.json').read_text());assert audit['status']=='static-checked-not-board-qualified'
 S=Path('/opt/opi-lab/source-validation-20261017/gguf-fresh/export/w8-fused2-c1280-t16-cap1-productwide/l2');D=R/'graph-k8-v3'
 binary=T/'runtime/down-capture1280-probe/tensor-probe-capture1280';bm=json.loads((binary.parent/'build.json').read_text());assert sha(binary)==bm['binary_sha256']
 sources={'probe':binary,'board_probe.py':R/'board_probe.py','original.nb':S/'export_nbg_unify/network_binary.nb','outlier8.nb':D/'export_nbg_unify/network_binary.nb','row-scale.nb':T/'graphs/down-row-scale-c1280-m16-l2/export_nbg_unify/network_binary.nb'}
 assert sha(sources['original.nb'])==audit['source_nb_sha256'];assert sha(sources['outlier8.nb'])==audit['candidate_nb_sha256']
 capture=json.loads((T/'results/down-capture-c1280-replay-r1/config.json').read_text());fixtures=[]
 for case in sorted(set(x['case'] for x in capture['fixtures'])):
  rows=sorted([x for x in capture['fixtures'] if x['case']==case],key=lambda x:x['position']);static=64 if case.startswith('s64-') else 128
  selected={0,16,static-16,static,static+16,rows[-1]['position']}
  fixtures.extend(x for x in rows if x['position'] in selected)
 assert len(fixtures)==12
 for f in fixtures:
  source=Path(f['input_path']);assert sha(source)==f['input_sha256'];sources[f['key']+'.f32']=source
 sequence=[]
 for variant,nb in [('original','original.nb'),('row-scale','row-scale.nb'),('outlier8','outlier8.nb'),('original-repeat','original.nb')]:
  for f in fixtures:sequence.append({**f,'run_id':f['key']+'-'+variant,'variant':variant,'capacity':1280,'query_tokens':16,'first_layer':2,'nb':remote+'/'+nb,'nb_sha256':sha(sources[nb]),'input':remote+'/'+f['key']+'.f32','expected_output_bytes':4*16*(896+4*128)})
 cfg={'run':'outlier8-20261026-r1','scope':'fused L2/L3 local mixed-precision screen; no full-model TTFT or semantic-quality claim','binary':remote+'/probe','binary_sha256':sha(binary),'warmup':3,'repeats':10,'fixtures':fixtures,'sequence':sequence,'audit_sha256':sha(R/'STATIC_AUDIT.json'),'source_sha256':{str(f.relative_to(T)):sha(f) for f in [Path(__file__),R/'board_probe.py',R/'build_k8_v3.py',R/'ANALYSIS.json',R/'STATIC_AUDIT.json']}}
 (R/'config.json').write_text(json.dumps(cfg,indent=2));sources['config.json']=R/'config.json';hashes={n:sha(f) for n,f in sources.items()};need=sum(f.stat().st_size for f in sources.values());state['upload_raw_bytes']=need;save()
 tarpath=R/'upload.tar.gz'
 with tarfile.open(tarpath,'w:gz',dereference=True) as tar:
  for n,f in sources.items():tar.add(f,arcname=n,recursive=False)
 archive_sha=sha(tarpath)
 with b.hostlock():
  health=locked('df -B1 --output=avail /tmp; cat /proc/meminfo; cat /sys/kernel/debug/viplite/mem_profile',capture_output=True,text=True,check=True).stdout
  (R/'pre-upload-health.txt').write_text(health);ls=health.splitlines();free=int(ls[1]);mem=int(next(x.split()[1] for x in ls if x.startswith('MemAvailable:')))*1024
  assert free>need+tarpath.stat().st_size+1100*1024**2,(free,need)
  assert mem>need+tarpath.stat().st_size+1300*1024**2,(mem,need)
  locked('mkdir '+remote,check=True);state.update(status='uploading',archive_sha256=archive_sha,archive_bytes=tarpath.stat().st_size);save()
  upload(tarpath,remote+'/upload.tar.gz')
  locked('sha256sum -c -',input=(archive_sha+'  '+remote+'/upload.tar.gz\n').encode(),capture_output=True,check=True)
  locked('tar -xzf '+remote+'/upload.tar.gz -C '+remote,check=True)
  locked('sha256sum -c -',input=''.join(h+'  '+remote+'/'+n+'\n' for n,h in hashes.items()).encode(),capture_output=True,check=True)
  locked('chmod 755 '+remote+'/probe',check=True)
  command='flock -w 5 /run/opi-npu-kv-benchmark.lock timeout -k 10 600 python3 -u '+remote+'/board_probe.py --config '+remote+'/config.json --out '+remote+'/result'
  wrapper=command+' >'+remote+'/process.log 2>&1; rc=$?; echo "$rc" >'+remote+'/exit'
  z=b.ssh('nohup sh -c '+shlex.quote(wrapper)+' </dev/null >/dev/null 2>&1 & echo $!',capture_output=True,text=True,check=True,timeout=20);state.update(status='running',board_pid=int(z.stdout.strip()));save()
  for i in range(130):
   time.sleep(5);z=b.ssh('cat '+remote+'/exit',capture_output=True,text=True,timeout=20)
   if z.returncode==0:state['returncode']=int(z.stdout.strip());break
  else:raise TimeoutError('observation timeout; do not duplicate board job')
  locked('cp '+remote+'/process.log '+remote+'/result/process.log',check=True)
  z=locked('tar -czf - -C '+remote+'/result .',capture_output=True,timeout=90,check=True);archive=R/'results.tar.gz';archive.write_bytes(z.stdout);extract(archive,R/'results')
  state.update(status='complete' if state['returncode']==0 else 'failed',ended=time.time(),result_archive_sha256=sha(archive));save()
except BaseException:state.update(status='failed',ended=time.time(),error=traceback.format_exc());save();raise
