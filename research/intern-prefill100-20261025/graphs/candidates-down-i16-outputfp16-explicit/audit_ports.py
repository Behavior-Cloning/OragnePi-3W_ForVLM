from pathlib import Path
import json,hashlib,fcntl,subprocess
D=Path(__file__).resolve().parent;R=D.parent.parent;O=D/'c512-m32-l2';S=D.parent/'cases/c512-m32-l2';M=D.parent/'down-minimal-outputfp16'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
state=json.loads((D/'status.json').read_text());assert state['status']=='compiled_pending_audit' and state['returncode']==0
sdk='/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools/vsimulator';ins=R/'graphs/candidates-w16-storage/inspect_nbg';ports={}
with open('/tmp/opi-prefill-compile.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 jobs=[('source',S),('candidate',O)]+[(n,M/n) for n in ['fc-i16-control','fc-fp16-output','conv-i16-control','conv-fp16-output']]
 for name,folder in jobs:
  nb=folder/'export_nbg_unify/network_binary.nb'
  cmd=['docker','run','--rm','--network','none','-v',str(R)+':'+str(R),'-e','LD_LIBRARY_PATH='+sdk+'/lib:'+sdk+'/../common/lib','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B','-e','VSIMULATOR_SHADER_CORE_COUNT=1','--entrypoint',str(ins),'ubuntu-npu:v2.0.10.2',str(nb)]
  r=subprocess.run(cmd,capture_output=True,text=True,timeout=30);assert r.returncode==0,r.stderr;ports[name]=json.loads(r.stdout);(D/(name+'-actual-ports.json')).write_text(r.stdout)
assert ports['source']==ports['candidate'] and ports['candidate']['count']==16
for op in ['fc','conv']:
 a,b=[ports[op+'-'+x] for x in ['i16-control','fp16-output']]
 assert a['count']==b['count']==2
 ai=[x for x in a['ports'] if x['direction']==0];bi=[x for x in b['ports'] if x['direction']==0];assert ai==bi and len(ai)==1 and ai[0]['dtype']==4
 ao=[x for x in a['ports'] if x['direction']==1];bo=[x for x in b['ports'] if x['direction']==1];assert len(ao)==len(bo)==1 and ao[0]['dims']==bo[0]['dims'] and ao[0]['dtype']==4 and bo[0]['dtype']==15
nb=O/'export_nbg_unify/network_binary.nb'
m={'status':'actual_external_ports_verified_not_board_validated','scope':'C512 M32 two-layer output-FP16 alternative, original full external contract preserved; minimal mixed FP16 output confirmed by importing actual binaries','nb_sha256':sha(nb),'nb_bytes':nb.stat().st_size,'actual_ports_exact':True,'minimal_four_queries_passed':True,'board_runs':0,'full_model_quality_verified':False,'goal_100ms_proved':False,'inspector_sha256':sha(ins),'audit_source_sha256':sha(Path(__file__))}
(D/'integration-manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m))
