from pathlib import Path
import json,yaml,numpy as np,shutil,hashlib,fcntl,subprocess,time,traceback
D=Path(__file__).resolve().parent;T=D.parent.parent
S=Path('/opt/opi-lab/source-validation-20261017/gguf-fresh/export/w8-fused2-c1280-t16-cap1-productwide/l2')
C=T/'graphs/candidates-down-i16-explicit/c512-m32-l2';O=D/'c1280-m16-l2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
state={'status':'preparing','started':time.time(),'scope':'C1280 M16 full-model integration candidate; same frozen I16 grids, not quality approved'}
def save():(D/'status.json').write_text(json.dumps(state,indent=2)+'\n')
try:
 save();assert not O.exists();O.mkdir()
 g=json.loads((S/'decoder.json').read_text());q=yaml.safe_load((S/'decoder.quantize').read_text());cq=yaml.safe_load((C/'decoder.quantize').read_text())
 data=np.load(S/'decoder.data',allow_pickle=True).item();cd=np.load(C/'decoder.data',allow_pickle=True).item()
 changes=[]
 for i in range(2):
  fc=f'b{i}_fullconnect_116';src=f'@b{i}_reshape_115:out0';cv=f'down_i16_{2000+i}';dest='@'+cv+':out0'
  assert np.array_equal(data[fc]['weight'],cd[fc]['weight'])
  assert q['quantize_parameters'][src]['qtype']=='float16' and g['Layers'][fc]['inputs']==[src]
  g['Layers'][cv]={'name':cv,'op':'dtype_converter','parameters':{},'inputs':[src],'outputs':['out0']};g['Layers'][fc]['inputs']=[dest]
  for key in [dest,'@'+fc+':weight','@'+fc+':out0']:
   changes.append({'key':key,'before':q['quantize_parameters'].get(key),'after':cq['quantize_parameters'][key]});q['quantize_parameters'][key]=cq['quantize_parameters'][key]
 (O/'decoder.json').write_text(json.dumps(g,indent=2)+'\n');(O/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False));(O/'decoder.data').symlink_to((S/'decoder.data').resolve())
 for name in ['tensor-inputmeta.yml','conversion.json']:shutil.copy2(S/name,O/name)
 (D/'pre-export-audit.json').write_text(json.dumps({'source':str(S),'source_graph_sha256':sha(S/'decoder.json'),'source_quant_sha256':sha(S/'decoder.quantize'),'source_data_sha256':sha(S/'decoder.data'),'grid_source_sha256':sha(C/'decoder.quantize'),'candidate_graph_sha256':sha(O/'decoder.json'),'candidate_quant_sha256':sha(O/'decoder.quantize'),'changed_quant_entries':changes,'down_source_weights_exact':True,'calibration':'frozen 24 positions single canonical trajectory, no refit on semantic evaluation','actual_ports_pending':True},indent=2)+'\n')
 lock=Path('/tmp/opi-prefill-compile.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert Path('/proc/539/cmdline').read_bytes()==b'sleep\x0021600\x00'
 source=S.parents[1]
 cmd=['docker','run','--rm','--name','opi-down-i16-c1280','--network','none','--cpus','4','--memory','8g','-e','TF_CPP_MIN_LOG_LEVEL=3','-e','OMP_NUM_THREADS=4','-e','VSIMULATOR_SHADER_CORE_COUNT=1','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B','-v',str(T)+':'+str(T),'-v',str(source)+':'+str(source)+':ro','-w',str(O),'ubuntu-npu:v2.0.10.2','python3','/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py','export','ovxlib','--model','decoder.json','--model-data','decoder.data','--model-quantize','decoder.quantize','--dtype','quantized','--with-input-meta','tensor-inputmeta.yml','--target-ide-project','linux64','--pack-nbg-unify','--optimize','VIP9000NANODI_PLUS_PID0X1000003B','--viv-sdk','/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools','--output-path','export/decoder']
 state.update(status='compiling',command=cmd);save()
 with (D/'export.log').open('wb') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=600)
 nb=O/'export_nbg_unify/network_binary.nb';state.update(status='compiled_pending_audit' if p.returncode==0 and nb.exists() else 'failed',returncode=p.returncode,nb_sha256=sha(nb) if nb.exists() else None,nb_bytes=nb.stat().st_size if nb.exists() else None,ended=time.time());save()
except BaseException:state.update(status='failed',error=traceback.format_exc(),ended=time.time());save();raise
