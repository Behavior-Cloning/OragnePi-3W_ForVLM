import sys,time,json,hashlib,subprocess,traceback
from pathlib import Path
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
p=Path(__file__).resolve().parent;ctx=p.parent/'intern-context256-20261025/full-stack256';opt=p.parent/'intern-nextopt-20261025';remote='/opt/opi-vlm/intern-candidates-20261025';base='/opt/opi-vlm/npu-kv-probe/wsl-fresh-peak155-20261018';state={'status':'waiting_for_ctx_stack','started':time.time()}
def save():(p/'deploy-status.json').write_text(json.dumps(state,indent=2))
save()
try:
 deadline=time.monotonic()+1800
 while True:
  s=json.loads((ctx/'status.json').read_text())
  if s['state']=='complete':break
  if s['state']=='failed':raise RuntimeError(s.get('error','ctx build failed'))
  if time.monotonic()>deadline:raise TimeoutError('ctx compilation deadline')
  time.sleep(10)
 cm=json.loads((ctx/'deployment-overlay.json').read_text());tm=json.loads((opt/'prefill8-stack-manifest.json').read_text());assert len(cm['replace'])==24 and len(tm['blocks'])==12
 spec={'base':base,'ctx':remote+'/ctx256','tail':remote+'/tail8','expected':{}}
 files={}
 for item in cm['replace']:
  name='ctx256/'+item['output_name'];files[name]=Path(item['candidate_nb']);spec['expected'][remote+'/'+name]=item['candidate_sha256']
 for item in s['cases']:spec['expected'][base+'/'+item['output_name']]=item['source_nb_sha256']
 for item in cm['reuse_unchanged']:
  if 'sha256' in item:spec['expected'][base+'/'+item['name']]=item['sha256']
 for item in tm['blocks']:
  name='tail8/prefill-layer'+str(item['first'])+'.nb';files[name]=Path(item['output'])/'export_nbg_unify/network_binary.nb';spec['expected'][remote+'/'+name]=item['binary_sha256']
 files['test-greedy-arm']=opt/'runtime-greedy/test-greedy-arm';spec['expected'][remote+'/test-greedy-arm']=hashlib.sha256(files['test-greedy-arm'].read_bytes()).hexdigest()
 (p/'deploy-spec.json').write_text(json.dumps(spec,indent=2));files['deploy-spec.json']=p/'deploy-spec.json';files['board_verify_candidates.py']=p/'board_verify_candidates.py'
 with b.hostlock():
  b.ssh('mkdir -p '+remote+'/ctx256 '+remote+'/tail8',check=True,timeout=20);state.update(status='copying',total=len(files),copied=0);save()
  for name,src in files.items():
   if remote+'/'+name in spec['expected']:assert hashlib.sha256(src.read_bytes()).hexdigest()==spec['expected'][remote+'/'+name]
   b.scp(str(src),b.HOST+':'+remote+'/'+name,check=True,timeout=90);state.update(copied=state['copied']+1,current=name);save()
  b.ssh('chmod 755 '+remote+'/test-greedy-arm',check=True,timeout=20);state['status']='verifying';save()
  z=b.ssh('flock -w 5 /run/opi-npu-kv-benchmark.lock python3 '+remote+'/board_verify_candidates.py',capture_output=True,text=True,timeout=150)
  (p/'deploy-verify.log').write_text(z.stdout+z.stderr);assert z.returncode==0,z.stderr[-1000:]
  b.scp(b.HOST+':'+remote+'/verified-graphs.json',str(p/'verified-graphs.json'),check=True,timeout=30)
 hashes=json.loads((p/'verified-graphs.json').read_text());cfg=json.loads((p/'config.profile.json').read_text());cfg['variants']=[]
 flags={'OPI_TTFT_TAIL_MIN':'8','OPI_GROUP_PREFILL':'1','OPI_MASK_TEMPLATE':'1'}
 definitions=[('mask-base','resident-mask',base,1280,flags,'base'),('context256','resident-mask',remote+'/ctx256',256,flags,'ctx'),('tail8-c1280','resident-tail8',base,1280,{**flags,'OPI_PREFILL8_DIR':remote+'/tail8'},'base'),('greedy-c1280','resident-greedy',base,1280,{**flags,'OPI_GREEDY_I16_SINGLE_PASS':'1'},'base'),('context256-greedy','resident-greedy',remote+'/ctx256',256,{**flags,'OPI_GREEDY_I16_SINGLE_PASS':'1'},'ctx'),('mask-repeat','resident-mask',base,1280,flags,'base')]
 for name,binary,graph,cap,env,key in definitions:
  expected=dict(hashes[key]);
  if name=='tail8-c1280':expected.update(hashes['tail'])
  cfg['variants'].append({'name':name,'binary':binary,'graph':graph,'capacity':cap,'env':env,'expected_graph_sha256':expected,'speed_profileflag':'none'})
 (p/'config.candidates.json').write_text(json.dumps(cfg,indent=2));state.update(status='ready',ended=time.time(),variants=len(cfg['variants']));save()
except BaseException:state.update(status='failed',error=traceback.format_exc(),ended=time.time());save();raise
