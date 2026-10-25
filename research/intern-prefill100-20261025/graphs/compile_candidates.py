#!/usr/bin/env python3
"""Frozen-precision C512/M16..256 fused-two-layer probes; no board calls."""
from pathlib import Path
import argparse,copy,fcntl,hashlib,json,os,shutil,subprocess,sys,time,traceback,uuid
import numpy as np,yaml
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[2]
SOURCE=Path('/opt/opi-lab/source-validation-20261017/gguf-fresh/export')
BASE=SOURCE/'w8-fused2-c1280-t16-cap1-productwide'
OLD=REPO/'research/intern-context256-20261025'
LIMIT=2*1024**3;RESERVE=8*1024**3
sys.path.insert(0,str(REPO/'npu/kv-cache'));from fused_contract import verify_contract
p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--tokens',nargs='+',type=int,default=[16,64,128,32,256]);p.add_argument('--layers',nargs='+',type=int,default=[0,2]);p.add_argument('--capacity',type=int,default=512);a=p.parse_args()
assert a.capacity==512 and all(t in [16,32,64,128,256] for t in a.tokens) and all(l in [0,2] for l in a.layers)
keepers=[p for p in Path('/proc').glob('[0-9]*/cmdline') if p.is_file() and p.read_bytes()==b'sleep\x0021600\x00']
assert keepers, 'explicit WSL sleep21600 keeper required before detached compilation'
lock=(ROOT/'.compile.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def write(p,d):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def diff(a,b,path=''):
 assert type(a)==type(b),path
 if isinstance(a,dict):
  assert a.keys()==b.keys(),path
  return sum([diff(a[k],b[k],path+'/'+k) for k in a],[])
 if isinstance(a,list):
  assert len(a)==len(b),path
  return sum([diff(x,y,path+'/'+str(i)) for i,(x,y) in enumerate(zip(a,b))],[])
 return [] if a==b else [{'path':path,'before':a,'after':b}]
def assign(g,path,before,after):
 parts=path.strip('/').split('/');node=g
 for part in parts[:-1]:node=node[int(part)] if isinstance(node,list) else node[part]
 k=int(parts[-1]) if isinstance(node,list) else parts[-1]
 assert node[k]==before,(path,node[k],before);node[k]=after

def disk():
 v=os.statvfs('/mnt/f');files=[p for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink()]
 return {'f_free_bytes':v.f_bavail*v.f_frsize,'new_regular_bytes':sum(p.stat().st_size for p in files)}
def guard():
 d=disk();assert d['f_free_bytes']>RESERVE and d['new_regular_bytes']<LIMIT-200*1024**2,d
state={'state':'preflight','started_unix':time.time(),'pid':os.getpid(),'execute':a.execute,'cases':[],'space_start':disk()}
def save():
 write(ROOT/'status.json',state)
 write(ROOT/'deployment-manifest.json',{'scope':'single fused2-layer probes only, not full-model qualification','capacity':a.capacity,'layers_covered':[0,1,2,3],'cases':[x for x in state['cases'] if x['state']=='compiled'],'compile_failures':[x for x in state['cases'] if x['state']!='compiled'],'whole_model_numeric_status':'pending; source/hash parity is not inference parity','runtime_contract':'11 F32 fixture input segments in conversion input_order; 5 output tensors; no board access performed'})
save()
try:
 mtemplate=json.loads((OLD/'prefill8-stack256/plan.json').read_text())['M_changes_from_c256_t16']
 ctemplate=next(x for x in json.loads((OLD/'graph-quantization-audit.json').read_text())['cases'] if x['name']=='prefill16')['graph_changes']
 assert len(mtemplate)==41 and len(ctemplate)==10
 image=subprocess.check_output(['docker','image','inspect','ubuntu-npu:v2.0.10.2','--format','{{.Id}}'],text=True).strip()
 assert image=='sha256:b2447e88c70d334cbadf91d1cfef2874d6a4360f0e2b075606df32547ec8cbe3'
 sources={};plans=[]
 evidence=REPO/'research/intern1b-operator-opt-20261023/results/20261023T064955/events.jsonl'
 known=json.loads(evidence.open().readline())['graph_sha256']
 for first in a.layers:
  src=BASE/f'l{first}';conv=json.loads((src/'conversion.json').read_text())
  assert conv['capacity']==1280 and conv['query_tokens']==16 and conv['first_layer']==first and conv['fusion_width']==2
  nb=src/'export_nbg_unify/network_binary.nb';assert sha(nb)==known[f'prefill-layer{first}.nb']
  hashes={n:sha(src/n) for n in ['decoder.json','decoder.data','decoder.quantize','tensor-inputmeta.yml','conversion.json','export/decoder.export.data']}
  weights=np.load(src/'decoder.data',allow_pickle=True).item();arrays=[]
  for node,vals in weights.items():
   for key,v in vals.items():
    v=np.asarray(v);arrays.append({'node':node,'field':key,'dtype':str(v.dtype),'shape':list(v.shape),'elements':int(v.size),'sha256':hashlib.sha256(v.tobytes(order='C')).hexdigest()})
  sources[first]={'path':str(src),'hashes':hashes,'data_canonical':str((src/'decoder.data').resolve()),'payload_canonical':str((src/'export/decoder.export.data').resolve()),'baseline_nb_sha256':sha(nb),'arrays':arrays}
 for tokens in a.tokens:
  for first in a.layers:
   src=Path(sources[first]['path']);original=json.loads((src/'decoder.json').read_text());target=copy.deepcopy(original)
   edits={}
   for ch in ctemplate:
    if 'mask_104/parameters/shape/3' in ch['path']:edits[ch['path']]={'path':ch['path'],'before':1296,'after':a.capacity+tokens}
    else:edits[ch['path']]={'path':ch['path'],'before':1280,'after':a.capacity}
   for ch in mtemplate:
    if 'mask_104/parameters/shape/3' in ch['path']:continue
    before=ch['before'];assert before in [16,112]
    edits[ch['path']]={'path':ch['path'],'before':before,'after':tokens if before==16 else 7*tokens}
   for ch in edits.values():assign(target,**ch)
   changes=diff(original,target);assert len(changes)==(10 if tokens==16 else 49)
   plans.append({'id':f'c{a.capacity}-m{tokens}-l{first}','first_layer':first,'query_tokens':tokens,'capacity':a.capacity,'source':sources[first],'changes':changes,'graph':target})
 write(ROOT/'source-provenance.json',{'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'sdk_image':image,'sdk':'ACUITY 6.30.22 / VivanteIDE5.11.0','chip':'VIP9000NANODI_PLUS_PID0X1000003B','baseline_board_evidence':str(evidence),'baseline_board_evidence_sha256':sha(evidence),'sources':sources,'frozen_precision':'original source quantize byte equal; original data same canonical file; compiled exported payload must be byte equal','capacity_basis':'system128+body256+dynamic_template8+output32 <=424; C512 leaves padding, no dynamic-token truncation'})
 write(ROOT/'plan.json',{'cases':[{k:v for k,v in x.items() if k!='graph'} for x in plans],'budget_bytes':LIMIT,'f_reserve_bytes':RESERVE,'sequential':True,'compile_timeout_seconds':600,'algorithm_changes':'capacity and query dimension only; unchanged graph topology/weights/quantization; no internal history crop in initial matrix','prefill_goal_ms':100,'ttft_goal_ms':100,'goal_status':'unmeasured, no claimed speedup'})
 state.update(state='preflight-passed',planned_cases=len(plans));save()
 if not a.execute:raise SystemExit(0)
 for item in plans:
  guard();name=item['id'];out=ROOT/'cases'/name;out.mkdir(parents=True,exist_ok=False)
  state.update(state='compiling',current=name);save();src=Path(item['source']['path']);start=time.monotonic()
  try:
   for n,h in item['source']['hashes'].items():assert sha(src/n)==h
   write(out/'decoder.json',item['graph']);(out/'decoder.data').symlink_to((src/'decoder.data').resolve());shutil.copy2(src/'decoder.quantize',out/'decoder.quantize')
   meta=yaml.safe_load((src/'tensor-inputmeta.yml').read_text())
   for port in meta['input_meta']['databases'][0]['ports']:port['shape']=item['graph']['Layers'][port['lid']]['parameters']['shape']
   (out/'tensor-inputmeta.yml').write_text(yaml.safe_dump(meta,sort_keys=False))
   write(out/'same-source-audit.json',{'source':item['source'],'graph_changes':item['changes'],'weight_arrays':item['source']['arrays'],'quantize_equal':sha(out/'decoder.quantize')==item['source']['hashes']['decoder.quantize'],'data_canonical_same':(out/'decoder.data').resolve()==(src/'decoder.data').resolve(),'target_shapes':{k:v['parameters']['shape'] for k,v in item['graph']['Layers'].items() if v['op']=='input'}})
   container='opi-prefill100-'+uuid.uuid4().hex[:10]
   cmd=['docker','run','--rm','--name',container,'--network','none','--cpus','4','--memory','8g','-e','TF_CPP_MIN_LOG_LEVEL=3','-e','OMP_NUM_THREADS=4','-e','VSIMULATOR_SHADER_CORE_COUNT=1','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B','-v',str(ROOT)+':'+str(ROOT),'-v',str(SOURCE)+':'+str(SOURCE)+':ro','-w',str(out),'ubuntu-npu:v2.0.10.2','python3','/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py','export','ovxlib','--model','decoder.json','--model-data','decoder.data','--model-quantize','decoder.quantize','--dtype','quantized','--with-input-meta','tensor-inputmeta.yml','--target-ide-project','linux64','--pack-nbg-unify','--optimize','VIP9000NANODI_PLUS_PID0X1000003B','--viv-sdk','/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools','--output-path','export/decoder']
   with (out/'export.log').open('w') as log:
    try:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=600).returncode
    except subprocess.TimeoutExpired:
     subprocess.run(['docker','stop','-t','2',container],stdout=log,stderr=subprocess.STDOUT,timeout=15);rc=124
   conv={'first_layer':item['first_layer'],'fusion_width':2,'capacity':a.capacity,'query_tokens':item['query_tokens'],'source':str(src),'source_hashes':item['source']['hashes'],'compile_command':cmd,'exit_code':rc,'seconds':time.monotonic()-start,'scope':'only C/M dimension edits; quantize/weights immutable'}
   write(out/'conversion.json',conv)
   if rc:raise RuntimeError(f'compiler rc {rc}')
   conv.update(verify_contract(out,2));write(out/'conversion.json',conv)
   for n,h in item['source']['hashes'].items():assert sha(src/n)==h
   assert sha(out/'decoder.quantize')==item['source']['hashes']['decoder.quantize']
   assert sha(out/'decoder.data')==item['source']['hashes']['decoder.data']
   payload=out/'export/decoder.export.data';actual=sha(payload);assert actual==item['source']['hashes']['export/decoder.export.data'],'compiled quantized payload mismatch'
   payload_bytes=payload.stat().st_size;payload.unlink();payload.symlink_to((src/'export/decoder.export.data').resolve())
   nb=out/'export_nbg_unify/network_binary.nb';row={'id':name,'state':'compiled','first_layer':item['first_layer'],'capacity':a.capacity,'query_tokens':item['query_tokens'],'nb':str(nb),'nb_sha256':sha(nb),'nb_bytes':nb.stat().st_size,'graph_sha256':sha(out/'decoder.json'),'quantize_sha256':sha(out/'decoder.quantize'),'payload_sha256':actual,'original_quantizer_and_exported_payload_byte_equal':True,'source_data_reference':str((out/'decoder.data').resolve()),'graph_changes':item['changes'],'input_order':conv['input_order'],'output_order':conv['output_order'],'compile_seconds':conv['seconds'],'dedup_payload_bytes':payload_bytes,'space':disk()}
  except Exception as e:
   row={'id':name,'state':'failed','error':repr(e),'traceback':traceback.format_exc(),'seconds':time.monotonic()-start,'log':str(out/'export.log')}
   write(out/'failure.json',row)
  state['cases'].append(row);save();print(json.dumps({k:v for k,v in row.items() if k in ['id','state','nb_bytes','compile_seconds','error']}),flush=True)
 state.update(state='complete' if all(x['state']=='compiled' for x in state['cases']) else 'complete-with-failures',ended_unix=time.time(),space_end=disk());save()
except SystemExit:raise
except BaseException as e:
 state.update(state='failed',error=repr(e),traceback=traceback.format_exc(),space_end=disk());save();raise
