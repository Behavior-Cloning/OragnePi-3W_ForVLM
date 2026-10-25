#!/usr/bin/env python3
"""Sequential full c256 overlay, each block from its current calibrated fused source.
No board access. Defaults to preflight only. --execute compiles after admission.
Weights remain source-readonly symlinks; no source graph/quantizer mutation.
"""
from pathlib import Path
import argparse,copy,hashlib,json,os,shutil,subprocess,sys,time,traceback,uuid
import numpy as np,yaml
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
SOURCE=Path('/opt/opi-lab/source-validation-20261017/gguf-fresh/export')
FULL=ROOT/'full-stack256';LIMIT=4*1024**3;RESERVE=8*1024**3
sys.path.insert(0,str(REPO/'npu/kv-cache'));from fused_contract import verify_contract
ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def disk():
 v=os.statvfs('/mnt/f');files=[p for p in FULL.rglob('*') if p.is_file() and not p.is_symlink()] if FULL.exists() else []
 return {'f_available_bytes':v.f_bavail*v.f_frsize,'full_regular_bytes':sum(p.stat().st_size for p in files),'full_allocated_bytes':sum(p.stat().st_blocks*512 for p in files)}
def guard():
 d=disk()
 if d['f_available_bytes']<RESERVE or d['full_regular_bytes']>LIMIT-256*1024**2:raise RuntimeError('space reserve/budget gate '+str(d))
 return d
def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
state={'state':'preflight','execute':a.execute,'started_unix':time.time(),'cases':[],'space_start':disk()}
FULL.mkdir(exist_ok=True)
def save():write(FULL/'status.json',state)
save()
try:
 admission=json.loads((ROOT/'single-block-admission.json').read_text());assert admission['admitted_for_full_stack_compile']
 audit=json.loads((ROOT/'graph-quantization-audit.json').read_text())
 templates={x['name']:x['graph_changes'] for x in audit['cases']}
 evidence=REPO/'research/intern1b-operator-opt-20261023/results/20261023T064955/events.jsonl'
 with evidence.open() as f:known=json.loads(f.readline())['graph_sha256']
 plans=[]
 for kind,variant,tokens in [('decode','w8-fused2-c1280-cap1-headmse',1),('prefill16','w8-fused2-c1280-t16-cap1-productwide',16)]:
  for first in range(0,24,2):
   src=SOURCE/variant/f'l{first}';c=json.loads((src/'conversion.json').read_text());assert c['first_layer']==first and c['fusion_width']==2 and c['capacity']==1280 and c['query_tokens']==tokens
   base_name=('' if kind=='decode' else 'prefill-')+f'layer{first}.nb'
   assert sha(src/'export_nbg_unify/network_binary.nb')==known[base_name],base_name
   old=json.loads((src/'decoder.json').read_text());g=copy.deepcopy(old)
   for change in templates[kind]:
    parts=change['path'].strip('/').split('/');node=g
    for part in parts[:-1]:node=node[int(part)] if isinstance(node,list) else node[part]
    key=int(parts[-1]) if isinstance(node,list) else parts[-1]
    assert node[key]==change['before'],(kind,first,change)
    node[key]=change['after']
   # Original source/quant/data retained; source hashes must still match at compilation.
   hashes={n:sha(src/n) for n in ['decoder.json','decoder.data','decoder.quantize','tensor-inputmeta.yml','conversion.json']}
   plans.append({'kind':kind,'first_layer':first,'query_tokens':tokens,'source':str(src),'output_name':base_name,'source_hashes':hashes,'source_nb_sha256':known[base_name],'capacity_changes':templates[kind],'graph':g})
 write(FULL/'plan.json',{'cases':[{k:v for k,v in x.items() if k!='graph'} for x in plans],'estimated_new_bytes_upper_bound':2*1024**3,'hard_budget_bytes':LIMIT,'sequential':True,'readonly_source_mount':str(SOURCE),'source_evidence':str(evidence),'scope':'24 graphs only; no copied heads or embeddings; maximum prompt200/output32','flags_unchanged':['batch16','OPI_TTFT_TAIL_MIN=8','OPI_GROUP_PREFILL=1','OPI_MASK_TEMPLATE=1']})
 state.update(state='preflight-passed',planned_graphs=len(plans));save()
 if not a.execute:raise SystemExit(0)
 for item in plans:
  guard();kind=item['kind'];first=item['first_layer'];src=Path(item['source']);out=FULL/'cases'/kind/f'l{first}';out.mkdir(parents=True,exist_ok=False)
  state.update(state='compiling',current=f'{kind}-l{first}');save()
  for name,h in item['source_hashes'].items():assert sha(src/name)==h
  write(out/'decoder.json',item['graph'])
  (out/'decoder.data').symlink_to(src/'decoder.data')
  shutil.copy2(src/'decoder.quantize',out/'decoder.quantize')
  meta=yaml.safe_load((src/'tensor-inputmeta.yml').read_text())
  for port in meta['input_meta']['databases'][0]['ports']:port['shape']=item['graph']['Layers'][port['lid']]['parameters']['shape']
  (out/'tensor-inputmeta.yml').write_text(yaml.safe_dump(meta,sort_keys=False))
  assert sha(out/'decoder.quantize')==item['source_hashes']['decoder.quantize']
  assert sha(out/'decoder.data')==item['source_hashes']['decoder.data']
  # Record every actual immutable array. Candidate path resolves to this same read-only file.
  weights=np.load(src/'decoder.data',allow_pickle=True).item();arrays=[]
  for node,values in weights.items():
   for key,value in values.items():
    v=np.asarray(value);arrays.append({'node':node,'parameter':key,'shape':list(v.shape),'dtype':str(v.dtype),'elements':int(v.size),'sha256':hashlib.sha256(v.tobytes(order='C')).hexdigest()})
  del weights
  write(out/'same-source-audit.json',{'source_hashes':item['source_hashes'],'source':str(src),'weight_reference_target':str((out/'decoder.data').resolve()),'weight_reference_is_exact_source':True,'quantizer_byte_equal':True,'capacity_changes':item['capacity_changes'],'arrays':arrays,'comparison_scope':'candidate weights are immutable same-file reference, not copied/requantized arrays'})
  c=json.loads((src/'conversion.json').read_text());c.update(capacity=256,original_capacity=1280,scope='current calibrated fused graph resized by audited capacity-only edit; immutable original data, exact quantizer; no recalibration',source_graph=str(src),source_graph_sha256=item['source_hashes'],capacity_changes=item['capacity_changes'])
  # Original layer0 single-block compiler already emitted exactly this graph and passed board smoke.
  prior=ROOT/'artifacts'/kind
  can_reuse=first==0 and json.loads((prior/'decoder.json').read_text())==item['graph'] and sha(prior/'decoder.quantize')==sha(out/'decoder.quantize') and sha(prior/'decoder.data')==sha(out/'decoder.data')
  started=time.monotonic()
  if can_reuse:
   (out/'export_nbg_unify').symlink_to(prior/'export_nbg_unify',target_is_directory=True)
   (out/'export').symlink_to(prior/'export',target_is_directory=True)
   rc=0;c['compile_reused_from']=str(prior);cmd=[]
  else:
   container='opi-ctx256-'+uuid.uuid4().hex[:10]
   cmd=['docker','run','--rm','--name',container,'--network','none','--cpus','4','--memory','8g','-e','TF_CPP_MIN_LOG_LEVEL=3','-e','OMP_NUM_THREADS=4','-e','VSIMULATOR_SHADER_CORE_COUNT=1','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B','-v',str(ROOT)+':/work','-v',str(SOURCE)+':'+str(SOURCE)+':ro','-w','/work/'+str(out.relative_to(ROOT)),'ubuntu-npu:v2.0.10.2','python3','/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py','export','ovxlib','--model','decoder.json','--model-data','decoder.data','--model-quantize','decoder.quantize','--dtype','quantized','--with-input-meta','tensor-inputmeta.yml','--target-ide-project','linux64','--pack-nbg-unify','--optimize','VIP9000NANODI_PLUS_PID0X1000003B','--viv-sdk','/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools','--output-path','export/decoder']
   with (out/'export.log').open('w') as log:
    try:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
    except subprocess.TimeoutExpired:
     subprocess.run(['docker','stop','-t','2',container],stdout=log,stderr=subprocess.STDOUT,timeout=15);rc=124
  c.update(exit_code=rc,seconds=time.monotonic()-started,compile_command=cmd)
  if rc==0:c.update(verify_contract(out,2))
  write(out/'conversion.json',c)
  if rc:raise RuntimeError('compile failed '+str(out))
  for name,h in item['source_hashes'].items():assert sha(src/name)==h,'source modified during compile'
  assert sha(out/'decoder.quantize')==item['source_hashes']['decoder.quantize'] and sha(out/'decoder.data')==item['source_hashes']['decoder.data']
  nb=out/'export_nbg_unify/network_binary.nb';entry={k:item[k] for k in ['kind','first_layer','query_tokens','source','source_nb_sha256','output_name']};entry.update(candidate_nb=str(nb),candidate_sha256=sha(nb),bytes=nb.stat().st_size,reused=can_reuse,compile_seconds=c['seconds'],source_hashes=item['source_hashes'],weight_arrays=len(arrays),weight_elements=sum(x['elements'] for x in arrays),space_after=disk())
  state['cases'].append(entry);save()
 # Runtime overlay references replacement blocks; heads/embedding stay at existing board graph directory.
 write(FULL/'deployment-overlay.json',{'capacity':256,'layers':24,'vocab':151674,'batch_tokens':16,'max_new_tokens':32,'base_board_directory':'/opt/opi-vlm/npu-kv-probe/wsl-fresh-peak155-20261018','replace':[{k:x[k] for k in ['output_name','candidate_nb','candidate_sha256','bytes']} for x in state['cases']],'reuse_unchanged':[{'name':f'layer{i}.nb','sha256':known[f'layer{i}.nb']} for i in range(24,28)]+[{'name':'embeddings.f32','note':'reuse actual base file; controller should record its SHA'}],'flags':['OPI_TTFT_TAIL_MIN=8','OPI_GROUP_PREFILL=1','OPI_MASK_TEMPLATE=1'],'precision_status':'all source quantizers/weights unchanged; whole-model NPU output parity not yet tested'})
 state.update(state='complete',ended_unix=time.time(),space_end=disk());assert state['space_end']['full_regular_bytes']<=LIMIT;save()
except SystemExit:raise
except BaseException as e:
 state.update(state='failed',error=repr(e),traceback=traceback.format_exc(),space_end=disk());save();raise
