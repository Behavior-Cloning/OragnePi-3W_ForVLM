import json,hashlib,sys,time,subprocess,shlex,traceback
from pathlib import Path
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
p=Path(__file__).resolve().parent;remote='/opt/opi-vlm/intern-candidates-20261025/tail8-c256';state={'status':'starting','started':time.time()}
def save():(p/'deploy-combined-status.json').write_text(json.dumps(state,indent=2))
save()
try:
 manifest=json.loads((p.parent/'intern-context256-20261025/prefill8-stack256/deployment-manifest.json').read_text());assert len(manifest['blocks'])==12 and manifest['capacity']==256
 expected={remote+'/'+v['output_name']:v['binary_sha256'] for v in manifest['blocks']}
 with b.hostlock():
  b.ssh('mkdir -p '+remote,check=True,timeout=20)
  for i,v in enumerate(manifest['blocks']):
   src=Path(v['candidate_nb']);assert hashlib.sha256(src.read_bytes()).hexdigest()==v['binary_sha256']
   b.scp(str(src),b.HOST+':'+remote+'/'+v['output_name'],check=True,timeout=90);state.update(status='copying',copied=i+1,total=12);save()
  state['status']='verifying';save()
  program='import json,hashlib;from pathlib import Path; paths='+repr(list(expected))+';print(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}))'
  z=b.ssh('flock -w 5 /run/opi-npu-kv-benchmark.lock python3 -c '+shlex.quote(program),capture_output=True,text=True,check=True,timeout=120)
  got=json.loads(z.stdout);assert got==expected
  (p/'verified-tail8-c256.json').write_text(json.dumps(got,indent=2))
 cfg=json.loads((p/'config.candidates.json').read_text());base=next(v for v in cfg['variants'] if v['name']=='context256-greedy');cfg['variants']=[]
 for name,tail in [('combined-base',False),('combined-tail8',True),('combined-repeat',False)]:
  v=json.loads(json.dumps(base));v.update(name=name,binary='resident-combined')
  if tail:v['env']['OPI_PREFILL8_DIR']=remote;v['expected_graph_sha256'].update(expected)
  cfg['variants'].append(v)
 cfg['all_cost_quality']=True;cfg['extra_diagnostic_case_ids']=['s64-d64-2']
 (p/'config.combined.json').write_text(json.dumps(cfg,indent=2))
 assert not (p/'combined-matrix-r1-controller.json').exists()
 log=(p/'combined-matrix-r1-controller.log').open('ab')
 child=subprocess.Popen([sys.executable,'-u',str(p/'controller_next.py'),'combined-matrix-r1','--config','config.combined.json','--repeats','2','--timeout','1500'],cwd=p,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 state.update(status='launched',controller_pid=child.pid,ended=time.time());save()
except BaseException:state.update(status='failed',error=traceback.format_exc(),ended=time.time());save();raise
