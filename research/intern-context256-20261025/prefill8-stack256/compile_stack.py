#!/usr/bin/env python3
"""Same-constant true-t8 c256 overlay. Sequential, no board calls.
Layer0 compile+payload audit gates compilation of layers2..22.
"""
from pathlib import Path
import argparse,copy,hashlib,json,os,shutil,subprocess,sys,time,traceback,uuid
import numpy as np,yaml
R=Path(__file__).resolve().parent;TASK=R.parent;REPO=TASK.parents[1]
SOURCE=Path('/opt/opi-lab/source-validation-20261017/gguf-fresh/export')
OLD8=Path('/opt/opi-lab/intern-nextopt-20261025/prefill8')
SRC256=TASK/'full-stack256/cases/prefill16';NEXT=REPO/'research/intern-nextopt-20261025'
sys.path.insert(0,str(REPO/'npu/kv-cache'));from fused_contract import verify_contract
p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def diff(a,b,path=''):
 assert type(a)==type(b),path
 if isinstance(a,dict):
  assert a.keys()==b.keys(),path
  return sum([diff(a[k],b[k],path+'/'+k) for k in a],[])
 if isinstance(a,list):
  assert len(a)==len(b),path
  return sum([diff(x,y,path+'/'+str(i)) for i,(x,y) in enumerate(zip(a,b))],[])
 return [] if a==b else [{'path':path,'before':a,'after':b}]
def apply(g,changes):
 out=copy.deepcopy(g)
 for c in changes:
  parts=c['path'].strip('/').split('/');node=out
  for part in parts[:-1]:node=node[int(part)] if isinstance(node,list) else node[part]
  k=int(parts[-1]) if isinstance(node,list) else parts[-1];assert node[k]==c['before'],c;node[k]=c['after']
 return out
def disk():
 v=os.statvfs('/mnt/f');files=[p for p in R.rglob('*') if p.is_file() and not p.is_symlink()]
 return {'f_available_bytes':v.f_bavail*v.f_frsize,'new_regular_bytes':sum(p.stat().st_size for p in files)}
def guard():
 d=disk();assert d['f_available_bytes']>8*1024**3 and d['new_regular_bytes']<2*1024**3,d
state={'state':'preflight','execute':a.execute,'started_unix':time.time(),'blocks':[],'space_start':disk()}
def save():write(R/'status.json',state)
save()
try:
 base0=SOURCE/'w8-fused2-c1280-t16-cap1-productwide/l0'
 g16=json.loads((base0/'decoder.json').read_text());g8=json.loads((OLD8/'l0/decoder.json').read_text())
 mchanges=diff(g16,g8);assert len(mchanges)==41
 for c in mchanges:
  assert '/parameters/' in c['path'] and (c['before'],c['after']) in [(16,8),(112,56),(1296,1288)],c
 # For c256 sources only the mask's total width changes from272 to264.
 m256=[dict(c,before=272,after=264) if (c['before'],c['after'])==(1296,1288) else c for c in mchanges]
 cap16=json.loads((TASK/'graph-quantization-audit.json').read_text())['cases'][1]['graph_changes']
 cap8=[dict(c,before=1288,after=264) if (c['before'],c['after'])==(1296,272) else c for c in cap16]
 source8={x['first']:x for x in json.loads((NEXT/'prefill8-stack-manifest.json').read_text())['blocks']}
 source256={x['first_layer']:x for x in json.loads((TASK/'full-stack256/full-audit.json').read_text())['rows'] if x['query_tokens']==16}
 plans=[]
 for first in range(0,24,2):
  src=SRC256/f'l{first}';old=OLD8/f'l{first}';conv=json.loads((src/'conversion.json').read_text())
  assert conv['capacity']==256 and conv['query_tokens']==16 and conv['first_layer']==first
  assert sha(src/'export_nbg_unify/network_binary.nb')==source256[first]['nb_sha256']
  assert sha(old/'export_nbg_unify/network_binary.nb')==source8[first]['binary_sha256']
  target=apply(json.loads((src/'decoder.json').read_text()),m256)
  independent=apply(json.loads((old/'decoder.json').read_text()),cap8)
  assert target==independent,'c256-t16 ->M8 differs from c1280-t8 ->C256'
  assert diff(json.loads((src/'decoder.json').read_text()),target)==m256
  fingerprints={name:sha(src/name) for name in ['decoder.json','decoder.quantize','decoder.data','export/decoder.export.data']}
  for name in ['decoder.quantize','decoder.data','export/decoder.export.data']:assert fingerprints[name]==sha(old/name),(first,name)
  plans.append({'first':first,'source_ctx256_t16':str(src),'reference_ctx1280_t8':str(old),'source_hashes':fingerprints,'source_t16_nb_sha256':source256[first]['nb_sha256'],'reference_t8_nb_sha256':source8[first]['binary_sha256'],'output_name':f'prefill-layer{first}.nb','graph':target})
 write(R/'plan.json',{'capacity':256,'query_tokens':8,'fusion_width':2,'M_changes_from_c256_t16':m256,'capacity_changes_from_c1280_t8':cap8,'two_construction_paths_identical':True,'blocks':[{k:v for k,v in x.items() if k!='graph'} for x in plans],'scope':'same calibrated source, no recalibration/quantization; true8 graph, not16-row padding','runtime':{'env':'OPI_PREFILL8_DIR=<board-t8-overlay>','base_capacity':256,'base_prefill_tokens':16,'max_new_tokens':32,'flags_preserve':['OPI_TTFT_TAIL_MIN=8','OPI_GROUP_PREFILL=1','OPI_MASK_TEMPLATE=1']}})
 state.update(state='preflight-passed',planned_blocks=12);save()
 if not a.execute:raise SystemExit(0)
 for item in plans:
  guard();first=item['first'];src=Path(item['source_ctx256_t16']);out=R/'cases'/f'l{first}';out.mkdir(parents=True,exist_ok=False)
  state.update(state='compiling',current_first=first);save()
  for name,h in item['source_hashes'].items():assert sha(src/name)==h
  write(out/'decoder.json',item['graph']);(out/'decoder.data').symlink_to((src/'decoder.data').resolve());shutil.copy2(src/'decoder.quantize',out/'decoder.quantize')
  meta=yaml.safe_load((src/'tensor-inputmeta.yml').read_text())
  for port in meta['input_meta']['databases'][0]['ports']:port['shape']=item['graph']['Layers'][port['lid']]['parameters']['shape']
  (out/'tensor-inputmeta.yml').write_text(yaml.safe_dump(meta,sort_keys=False))
  weights=np.load(src/'decoder.data',allow_pickle=True).item();arrays=[]
  for node,values in weights.items():
   for key,value in values.items():
    v=np.asarray(value);arrays.append({'node':node,'field':key,'dtype':str(v.dtype),'shape':list(v.shape),'elements':int(v.size),'sha256':hashlib.sha256(v.tobytes()).hexdigest()})
  del weights
  write(out/'same-source-audit.json',{'source_hashes':item['source_hashes'],'source_ctx256_t16':str(src),'weight_reference_target':str((out/'decoder.data').resolve()),'quantizer_byte_equal':sha(out/'decoder.quantize')==item['source_hashes']['decoder.quantize'],'weights_same_file_reference':True,'arrays':arrays,'M_changes':m256,'independent_C256_T8_graph_identical':True})
  container='opi-c256-t8-'+uuid.uuid4().hex[:10]
  cmd=['docker','run','--rm','--name',container,'--network','none','--cpus','4','--memory','8g','-e','TF_CPP_MIN_LOG_LEVEL=3','-e','OMP_NUM_THREADS=4','-e','VSIMULATOR_SHADER_CORE_COUNT=1','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B','-v',str(TASK)+':'+str(TASK),'-v',str(SOURCE)+':'+str(SOURCE)+':ro','-w',str(out),'ubuntu-npu:v2.0.10.2','python3','/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py','export','ovxlib','--model','decoder.json','--model-data','decoder.data','--model-quantize','decoder.quantize','--dtype','quantized','--with-input-meta','tensor-inputmeta.yml','--target-ide-project','linux64','--pack-nbg-unify','--optimize','VIP9000NANODI_PLUS_PID0X1000003B','--viv-sdk','/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools','--output-path','export/decoder']
  begin=time.monotonic()
  with (out/'export.log').open('w') as log:
   try:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
   except subprocess.TimeoutExpired:subprocess.run(['docker','stop','-t','2',container],stdout=log,stderr=subprocess.STDOUT,timeout=15);rc=124
  conv=json.loads((src/'conversion.json').read_text());conv.update(query_tokens=8,capacity=256,source_ctx256_t16=str(src),source_ctx1280_t8=item['reference_ctx1280_t8'],M_changes=m256,scope='same calibrated c256 graph, true batch8 shape only; retained historical conversion fields describe original source',compile_command=cmd,exit_code=rc,seconds=time.monotonic()-begin)
  if rc==0:conv.update(verify_contract(out,2))
  write(out/'conversion.json',conv)
  if rc:raise RuntimeError(f'compile failed l{first} rc{rc}')
  for name,h in item['source_hashes'].items():assert sha(src/name)==h
  assert sha(out/'decoder.quantize')==item['source_hashes']['decoder.quantize']
  assert sha(out/'decoder.data')==item['source_hashes']['decoder.data']
  payload=out/'export/decoder.export.data';assert sha(payload)==item['source_hashes']['export/decoder.export.data'],'actual quantized payload differs'
  # Drop only own duplicate payload after exact SHA proof; immutable baseline reused.
  payload_bytes=payload.stat().st_size;payload.unlink();payload.symlink_to((src/'export/decoder.export.data').resolve())
  nb=out/'export_nbg_unify/network_binary.nb';row={k:v for k,v in item.items() if k!='graph'};row.update(candidate_nb=str(nb),binary_sha256=sha(nb),binary_bytes=nb.stat().st_size,query_tokens=8,capacity=256,arrays=len(arrays),elements=sum(x['elements'] for x in arrays),quantizer_byte_equal=True,actual_quantized_payload_byte_equal=True,verified_payload_symlink_bytes_saved=payload_bytes,seconds=conv['seconds'],space_after=disk())
  state['blocks'].append(row)
  if first==0:state['first_block_compile_audit_passed']=True
  else:assert state.get('first_block_compile_audit_passed')
  save()
 write(R/'deployment-manifest.json',{'capacity':256,'tokens':8,'fusion_width':2,'runtime_env':'OPI_PREFILL8_DIR=<separate board directory containing prefill-layer0.nb ... prefill-layer22.nb>','base_graph_capacity':256,'base_prefill_tokens':16,'max_new_tokens':32,'blocks':state['blocks'],'all_binary_bytes':sum(x['binary_bytes'] for x in state['blocks']),'heads_embeddings_decode_prefill16':'reuse existing c256 full overlay; only add t8 bank','whole_model_numeric_status':'pending parent board validation; graph/constants audits do not substitute output parity'})
 state.update(state='complete',ended_unix=time.time(),space_end=disk());save()
except SystemExit:raise
except BaseException as e:state.update(state='failed',error=repr(e),traceback=traceback.format_exc(),space_end=disk());save();raise
