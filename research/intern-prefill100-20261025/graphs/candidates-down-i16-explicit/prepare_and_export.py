from pathlib import Path
import json,yaml,numpy as np,torch,sys,hashlib,math,shutil,time,fcntl,subprocess,traceback
D=Path(__file__).resolve().parent;T=D.parent.parent;S=T/'graphs/cases/c512-m32-l2';O=D/'c512-m32-l2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(): (D/'status.json').write_text(json.dumps(state,indent=2)+'\n')
state={'status':'preparing','started':time.time(),'scope':'independent I16 down weight/input candidate; single canonical calibration, not full-model quality'};save()
try:
 assert not O.exists();O.mkdir()
 torch.set_num_threads(2)
 sys.path.insert(0,str(T/'graphs/candidates-fp16'));from graph_reference_conv import evaluate
 g=json.loads((S/'decoder.json').read_text());data=np.load(S/'decoder.data',allow_pickle=True).item();q=yaml.safe_load((S/'decoder.quantize').read_text())
 meta=json.loads((T/'fixtures-canonical/c512-m32-l2/probe-fixture.json').read_text());raw=np.fromfile(T/'fixtures-canonical/c512-m32-l2/probe-input.f32',np.float32);feeds={};offset=0
 for key,n in meta['inputs']:
  node=next(v for v in g['Layers'].values() if v.get('op')=='input' and v['name']==key);feeds[key]=torch.from_numpy(raw[offset:offset+n].copy().reshape(node['parameters']['shape']));offset+=n
 assert offset==raw.size
 with torch.inference_mode(): outputs,cache=evaluate(g,data,feeds,return_intermediates=True)
 changes=[]
 for i in range(2):
  name=f'b{i}_fullconnect_116';ain=f'b{i}_reshape_115';x=cache[ain].numpy().reshape(32,-1);w=data[name]['weight'];assert x.shape==(32,4864) and w.shape==(4864,896)
  for key,values,kind in [(f'@{ain}:out0',x[:24],'activation'),(f'@{name}:out0',cache[name].numpy().reshape(32,-1)[:24],'output'),(f'@{name}:weight',w,'weight')]:
   maxabs=float(np.max(abs(values)));guard=1.25 if kind!='weight' else 1.0;fl=math.floor(math.log2(32767/(maxabs*guard)))
   new={'qtype':'i16','quantizer':'dynamic_fixed_point','rounding':'rtne','max_value':maxabs*guard,'min_value':-maxabs*guard,'fl':fl}
   old=q['quantize_parameters'][key];assert old['qtype']=='float16';q['quantize_parameters'][key]=new
   evalv=x[24:] if kind=='activation' else (cache[name].numpy().reshape(32,-1)[24:] if kind=='output' else w);codes=np.rint(evalv.astype(np.float64)*2**fl);sat=int(np.sum((codes>32767)|(codes<-32768)));recon=np.clip(codes,-32768,32767)/(2**fl)
   changes.append({'key':key,'before':old,'after':new,'kind':kind,'calibration_rows':'0:24 of one canonical trajectory' if kind!='weight' else 'all source FP32 weights','evaluation_rows':'24:32 same correlated trajectory; not independent quality' if kind!='weight' else 'all weights','evaluation_saturation':sat,'evaluation_relative_l2':float(np.linalg.norm(recon-evalv)/max(np.linalg.norm(evalv),1e-30))})
 for i in range(2):
  source=f'@b{i}_reshape_115:out0';key=f'down_i16_{2000+i}';target='@'+key+':out0'
  q['quantize_parameters'][target]=q['quantize_parameters'][source];q['quantize_parameters'][source]={'qtype':'float16','quantizer':'float16'}
  g['Layers'][key]={'name':key,'op':'dtype_converter','parameters':{},'inputs':[source],'outputs':['out0']}
  assert g['Layers'][f'b{i}_fullconnect_116']['inputs']==[source];g['Layers'][f'b{i}_fullconnect_116']['inputs']=[target]
 (O/'decoder.json').write_text(json.dumps(g,indent=2)+'\n');(O/'decoder.data').symlink_to((S/'decoder.data').resolve());shutil.copy2(S/'tensor-inputmeta.yml',O/'tensor-inputmeta.yml');shutil.copy2(S/'conversion.json',O/'conversion.json');(O/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False))
 (D/'candidate-audit.json').write_text(json.dumps({'scope':state['scope'],'source_graph_sha256':sha(S/'decoder.json'),'candidate_graph_sha256':sha(O/'decoder.json'),'source_data_sha256':sha(S/'decoder.data'),'source_quant_sha256':sha(S/'decoder.quantize'),'candidate_quant_sha256':sha(O/'decoder.quantize'),'changes':changes,'explicit_input_converters':['down_i16_2000','down_i16_2001'],'reshape_input_precision_restored':'float16','calibration_changes_note':'activation entries apply to added converter outputs, not reshapes in final graph','unchanged_external_contract_expected':True,'external_actual_ports_pending':True,'activation_basis':'FP32 source graph intermediate; does not simulate upstream NPU quantization','temperature_or_speed_claim':None},indent=2)+'\n')
 assert len(changes)==6
 state.update(status='prepared',changes=changes);save()
 lock=Path('/tmp/opi-prefill-compile.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert any(p.read_bytes()==b'sleep\x0021600\x00' for p in Path('/proc').glob('[0-9]*/cmdline') if p.exists())
 source=Path('/opt/opi-lab/source-validation-20261017/gguf-fresh/export')
 cmd=['docker','run','--rm','--name','opi-down-i16-explicit-screen','--network','none','--cpus','4','--memory','8g','-e','TF_CPP_MIN_LOG_LEVEL=3','-e','OMP_NUM_THREADS=4','-e','VSIMULATOR_SHADER_CORE_COUNT=1','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B','-v',str(T)+':'+str(T),'-v',str(source)+':'+str(source)+':ro','-w',str(O),'ubuntu-npu:v2.0.10.2','python3','/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py','export','ovxlib','--model','decoder.json','--model-data','decoder.data','--model-quantize','decoder.quantize','--dtype','quantized','--with-input-meta','tensor-inputmeta.yml','--target-ide-project','linux64','--pack-nbg-unify','--optimize','VIP9000NANODI_PLUS_PID0X1000003B','--viv-sdk','/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools','--output-path','export/decoder']
 state.update(status='compiling',command=cmd);save()
 with (D/'export.log').open('wb') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=600)
 nb=O/'export_nbg_unify/network_binary.nb';state.update(status='compiled_pending_port_and_numeric_audit' if r.returncode==0 and nb.exists() else 'export_failed',returncode=r.returncode,ended=time.time(),nb_sha256=sha(nb) if nb.exists() else None,nb_bytes=nb.stat().st_size if nb.exists() else None);save()
except BaseException:
 state.update(status='failed',error=traceback.format_exc(),ended=time.time());save();raise
