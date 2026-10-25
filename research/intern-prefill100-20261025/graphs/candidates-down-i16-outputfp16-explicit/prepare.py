from pathlib import Path
import json,yaml,hashlib,fcntl,subprocess,time,traceback,os,copy
D=Path(__file__).resolve().parent;T=D.parent.parent;S=D.parent/'candidates-down-i16-range16/c512-m32-l2';O=D/'c512-m32-l2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save():(D/'status.json').write_text(json.dumps(state,indent=2)+'\n')
state={'status':'preparing','started':time.time(),'pid':os.getpid(),'scope':'two-layer I16 down with explicit input conversion and FP16 output; unqualified new quantization candidate'};save()
try:
 minimal=json.loads((D.parent/'down-minimal-outputfp16/status.json').read_text());assert minimal['status']=='complete' and all(c['returncode']==0 and c['nb_bytes'] for c in minimal['cases'])
 lock=Path('/tmp/opi-prefill-compile.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not O.exists();O.mkdir()
 for n in ['decoder.json','tensor-inputmeta.yml','conversion.json']:(O/n).write_bytes((S/n).read_bytes())
 (O/'decoder.data').symlink_to((S/'decoder.data').resolve())
 q=yaml.safe_load((S/'decoder.quantize').read_text());before=copy.deepcopy(q);changes=[]
 for i in range(2):
  key=f'@b{i}_fullconnect_116:out0';assert q['quantize_parameters'][key]['qtype']=='i16'
  changes.append({'key':key,'before':q['quantize_parameters'][key],'after':{'qtype':'float16','quantizer':'float16'}});q['quantize_parameters'][key]={'qtype':'float16','quantizer':'float16'}
 assert {k for k in q['quantize_parameters'] if q['quantize_parameters'][k]!=before['quantize_parameters'][k]}=={x['key'] for x in changes}
 (O/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False))
 (D/'pre-export-audit.json').write_text(json.dumps({'source':str(S),'changes':changes,'same_graph':sha(O/'decoder.json')==sha(S/'decoder.json'),'same_source_payload':sha(O/'decoder.data')==sha(S/'decoder.data'),'input_ranges':'unchanged from range16: b0 fl12 (~8), b1 fl11 (~16); unsafe at first system chunk; not fixed by FP16 output','weight_grids':'unchanged I16 fl16/fl15; not original FP16 weight precision','actual_payload_audit_pending':True,'actual_ports_pending':True},indent=2)+'\n')
 cmd=json.loads((S.parent/'status.json').read_text())['command'];cmd=[x.replace(str(S),str(O)) for x in cmd];cmd[cmd.index('--name')+1]='opi-down-i16-outfp16-explicit'
 state.update(status='compiling',command=cmd);save()
 with (D/'export.log').open('wb') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=600)
 nb=O/'export_nbg_unify/network_binary.nb';state.update(status='compiled_pending_audit' if r.returncode==0 and nb.exists() else 'export_failed',returncode=r.returncode,ended=time.time(),nb_sha256=sha(nb) if nb.exists() else None,nb_bytes=nb.stat().st_size if nb.exists() else None);save()
except BaseException:state.update(status='failed',error=traceback.format_exc(),ended=time.time());save();raise
