#!/usr/bin/env python3
"""Compile only same-quantization Intern layers 0/1 at capacity256.
No board access. Sequential Docker jobs; snapshot source + emitted graphs.
"""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,time,traceback
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
SOURCE=Path('/opt/opi-lab/source-validation-20261017/gguf-fresh/export')
PY=Path('/opt/opi-lab/export-venv/bin/python')
COMPILER=REPO/'npu/kv-cache/compile_fused_decoder.py'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def snapshot():
 s=os.statvfs('/mnt/f')
 return {'unix':time.time(),'f_available_bytes':s.f_bavail*s.f_frsize,'task_file_bytes':sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())}
def save(state):
 (ROOT/'build-status.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
state={'state':'preparing','start':snapshot(),'sequential':True,'cases':[]}
save(state)
try:
 if state['start']['f_available_bytes']<8*1024**3:raise RuntimeError('F physical free below 8 GiB reserve')
 src=ROOT/'source'
 src.mkdir(exist_ok=True)
 copied=[]
 for rel in ['trained-manifest.json']+[f'w8-balanced/l{l}/{f}' for l in [0,1] for f in ['decoder.json','decoder.data','decoder.quantize','tensor-inputmeta.yml']]:
  p=SOURCE/rel;q=src/rel;q.parent.mkdir(parents=True,exist_ok=True)
  expected=sha(p)
  if not q.exists():shutil.copy2(p,q)
  assert sha(q)==expected,rel
  copied.append({'source':str(p),'copy':str(q.relative_to(ROOT)),'bytes':q.stat().st_size,'sha256':expected})
 tools={str(p.relative_to(REPO)):sha(p) for p in (REPO/'npu/kv-cache').glob('*.py')}
 provenance={'created_unix':time.time(),'source_copies':copied,'tool_hashes':tools,'image_id':subprocess.check_output(['docker','image','inspect','ubuntu-npu:v2.0.10.2','--format','{{.Id}}'],text=True).strip(),'scope':'layers0+1 only, shape capacity1280 to256, no recalibration, no board execution','max_prompt_tokens':200,'max_generated_tokens_current_test':32,'max_used_sequence_positions':232}
 (ROOT/'source-provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
 for name,tokens,original in [('decode',1,'w8-fused2-c1280-cap1-headmse'),('prefill16',16,'w8-fused2-c1280-t16-cap1-productwide')]:
  out=ROOT/'artifacts'/name
  if out.exists():raise RuntimeError('Refusing to overwrite existing output '+str(out))
  cmd=[str(PY),str(COMPILER),'--source',str(src),'--results-root',str(ROOT),'--variant','w8-balanced','--first','0','--width','2','--output',str(out),'--timeout','300','--compact-gqa','--capacity','256','--tokens',str(tokens),'--split-kv-attention','--split-softmax-i16','--split-current-matmul','--fc-weight-dfp','gateup','--fc-weight-dfp-method','mse','--fc-weight-dfp-max-extra-bits','1']
  state.update(state='compiling',current=name);save(state)
  started=time.monotonic()
  with (ROOT/(name+'-build.log')).open('w') as log:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=420).returncode
  case={'name':name,'query_tokens':tokens,'command':cmd,'returncode':rc,'wall_seconds':time.monotonic()-started,'original_graph':str(SOURCE/original/'l0'),'output':str(out),'after':snapshot()}
  state['cases'].append(case);save(state)
  if rc:raise RuntimeError(name+' compile failed rc='+str(rc))
 state.update(state='compiled',end=snapshot());save(state)
except BaseException as e:
 state.update(state='failed',error=repr(e),traceback=traceback.format_exc(),end=snapshot());save(state)
 raise
