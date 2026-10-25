from pathlib import Path
import json,hashlib,fcntl,subprocess
D=Path(__file__).resolve().parent;R=D.parent.parent;O=D/'c1280-m16-l2';S=Path('/opt/opi-lab/source-validation-20261017/gguf-fresh/export/w8-fused2-c1280-t16-cap1-productwide/l2')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
state=json.loads((D/'status.json').read_text());assert state['status']=='compiled_pending_audit' and state['returncode']==0
old=R/'graphs/candidates-down-i16-c1280/c1280-m16-l2';assert sha(O/'export/decoder.export.data')==sha(old/'export/decoder.export.data')
sdk='/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools/vsimulator';ins=R/'graphs/candidates-w16-storage/inspect_nbg';ports={}
with open('/tmp/opi-prefill-compile.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for name,nb in [('source',S/'export_nbg_unify/network_binary.nb'),('candidate',O/'export_nbg_unify/network_binary.nb')]:
  cmd=['docker','run','--rm','--network','none','-v',str(R)+':'+str(R),'-v',str(S)+':'+str(S)+':ro','-e','LD_LIBRARY_PATH='+sdk+'/lib:'+sdk+'/../common/lib','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B','-e','VSIMULATOR_SHADER_CORE_COUNT=1','--entrypoint',str(ins),'ubuntu-npu:v2.0.10.2',str(nb)]
  p=subprocess.run(cmd,capture_output=True,text=True,timeout=30);assert p.returncode==0,p.stderr;ports[name]=json.loads(p.stdout);(D/(name+'-actual-ports.json')).write_text(p.stdout)
assert ports['source']==ports['candidate'] and ports['candidate']['count']==16
nb=O/'export_nbg_unify/network_binary.nb';m={'status':'compiled_actual_ports_verified_not_board_validated','scope':'C1280 M16 layers2/3 full-model integration candidate; source original16 external ports exact; range16 repair from development screen','nb_path':str(nb),'nb_sha256':sha(nb),'nb_bytes':nb.stat().st_size,'source_nb_path':str(S/'export_nbg_unify/network_binary.nb'),'source_nb_sha256':sha(S/'export_nbg_unify/network_binary.nb'),'actual_ports_exact':True,'payload_equal_prior_c1280_i16':True,'payload_sha256':sha(O/'export/decoder.export.data'),'quant_grid_origin':'C512 range16 candidate; no further fit','board_runs':0,'full_model_quality_verified':False,'goal_100ms_proved':False,'inspector_sha256':sha(ins),'audit_source_sha256':sha(Path(__file__))}
(D/'integration-manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=2))
