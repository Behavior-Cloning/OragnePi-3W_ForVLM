import sys,json,subprocess,time,hashlib,tarfile,traceback
from pathlib import Path
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
p=Path(__file__).resolve().parent;remote='/opt/opi-vlm/graph-probes-20261025';opt=p.parent/'intern-nextopt-20261025';ctx=p.parent/'intern-context256-20261025';base='/opt/opi-vlm/npu-kv-probe/wsl-fresh-peak155-20261018'
status={'status':'preparing','started':time.time()}
def save():(p/'graph-probes-controller.json').write_text(json.dumps(status,indent=2))
save();sources={'board_graph_probes.py':p/'board_graph_probes.py','tensor8':opt/'tensor-probe','tensorctx':ctx/'tensor-probe-arm64','t8.nb':Path('/opt/opi-lab/intern-nextopt-20261025/prefill8/l0/export_nbg_unify/network_binary.nb'),'ctx-decode.nb':ctx/'artifacts/decode/export_nbg_unify/network_binary.nb','ctx-prefill16.nb':ctx/'artifacts/prefill16/export_nbg_unify/network_binary.nb'}
spec={'jobs':[],'hashes':{base+'/layer0.nb':'f6dc56df18b2537893e9089d79d9d6928fa0339d3ae485e450bb9a951cf63365',base+'/prefill-layer0.nb':'44271ccc82b1ac33037c25994f101966c6493041a60264ba705fa8ff5def6e4a'}}
for case in ['s64-d32-0','s64-d64-0','s128-d32-0','s128-d64-0']:
 for v in ['t8','t16']:sources[case+'-'+v+'.f32']=Path('/opt/opi-lab/intern-nextopt-20261025/fixtures')/case/v/'input.f32'
 for label in ['t16','t8','t16-repeat']:
  t='t8' if label=='t8' else 't16';spec['jobs'].append({'name':'prefill8/'+case+'/'+label,'probe':remote+'/tensor8','graph':remote+'/t8.nb' if t=='t8' else base+'/prefill-layer0.nb','input':remote+'/'+case+'-'+t+'.f32','repeats':50})
for mode in ['decode','prefill16']:
 for cap in [1280,256]:sources['ctx-'+str(cap)+'-'+mode+'.f32']=ctx/'fixtures'/('cap'+str(cap)+'-'+mode)/'probe-input.f32'
 for label in ['1280','256','1280-repeat']:
  cap='256' if label=='256' else '1280';spec['jobs'].append({'name':'context256/'+mode+'/'+label,'probe':remote+'/tensorctx','graph':remote+'/ctx-'+mode+'.nb' if cap=='256' else base+('/layer0.nb' if mode=='decode' else '/prefill-layer0.nb'),'input':remote+'/ctx-'+cap+'-'+mode+'.f32','repeats':30})
for name,src in sources.items():spec['hashes'][remote+'/'+name]=hashlib.sha256(src.read_bytes()).hexdigest()
(p/'graph-probes-spec.json').write_text(json.dumps(spec,indent=2));sources['spec.json']=p/'graph-probes-spec.json'
try:
 with b.hostlock():
  b.ssh('mkdir '+remote,check=True,timeout=20)
  for name,src in sources.items():b.scp(str(src),b.HOST+':'+remote+'/'+name,check=True,timeout=60)
  b.ssh('chmod 755 '+remote+'/tensor8 '+remote+'/tensorctx',check=True,timeout=20)
  status['status']='running';save()
  z=b.ssh('flock -w 5 /run/opi-npu-kv-benchmark.lock timeout -k 15 700 python3 '+remote+'/board_graph_probes.py',capture_output=True,text=True,timeout=730);status['returncode']=z.returncode;status['stderr']=z.stderr[-3000:];save()
  b.ssh('tar -czf '+remote+'/results.tar.gz -C '+remote+'/results .',check=True,timeout=60)
  arc=p/'graph-probes.tar.gz';b.scp(b.HOST+':'+remote+'/results.tar.gz',str(arc),check=True,timeout=120)
  out=p/'results/graph-probes';out.mkdir(parents=True,exist_ok=False)
  with tarfile.open(arc) as tar:
   for m in tar.getmembers():
    q=(out/m.name).resolve();assert q==out.resolve() or out.resolve() in q.parents;assert not m.issym() and not m.islnk()
   tar.extractall(out)
  status['status']='finished' if z.returncode==0 and (out/'complete.json').exists() else 'failed';status['ended']=time.time();save()
except BaseException:status.update(status='failed',error=traceback.format_exc(),ended=time.time());save();raise
