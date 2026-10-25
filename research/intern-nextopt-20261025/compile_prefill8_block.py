from pathlib import Path
import os,json,subprocess,time,hashlib
import numpy as np,yaml
ROOT=Path(__file__).resolve().parent
BUILD=Path('/opt/opi-lab/intern-nextopt-20261025')
BASE=Path('/opt/opi-lab/source-validation-20261017/gguf-fresh/export')
SRC=BASE/'w8-fused2-c1280-t16-cap1-productwide/l0'
OUT=BUILD/'prefill8/l0'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def freef():
 s=os.statvfs('/mnt/f');return s.f_bavail*s.f_frsize
assert freef()>8*1024**3
assert not OUT.exists()
BUILD.mkdir(parents=True,exist_ok=True)
command=['/opt/opi-lab/export-venv/bin/python',str(ROOT/'compiler-source/compile_fused_decoder.py'),'--source',str(BASE),'--variant','w8-balanced','--results-root','/opt/opi-lab','--first','0','--width','2','--output',str(OUT),'--timeout','300','--compact-gqa','--capacity','1280','--tokens','8','--fc-weight-dfp','gateup','--fc-weight-dfp-method','mse','--fc-weight-dfp-max-extra-bits','1','--split-kv-attention','--split-softmax-i16','--split-current-matmul']
status={'phase':'compiling','command':command,'source_t16':str(SRC),'output':str(OUT),'f_free_before':freef(),'started_unix':time.time()}
(ROOT/'prefill8-build-status.json').write_text(json.dumps(status,indent=2)+'\n')
t=time.monotonic()
with (ROOT/'prefill8-compile.log').open('w') as log:
 r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=360)
status.update(returncode=r.returncode,seconds=time.monotonic()-t,f_free_after=freef())
if r.returncode==0:
 old=yaml.safe_load((SRC/'decoder.quantize').read_text());new=yaml.safe_load((OUT/'decoder.quantize').read_text())
 status['all_quantize_parameters_identical']=old==new
 assert old==new,'t8 quantization changed'
 before=np.load(SRC/'decoder.data',allow_pickle=True).item();after=np.load(OUT/'decoder.data',allow_pickle=True).item()
 records=[]
 for name,items in before.items():
  if 'fullconnect' not in name or 'weight' not in items:continue
  for key in ['weight','bias']:
   if key not in items:continue
   a,b=items[key],after[name][key];assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
   records.append({'node':name,'field':key,'shape':list(a.shape),'sha256':hashlib.sha256(a.tobytes()).hexdigest()})
 assert len([x for x in records if x['field']=='weight'])==14
 nbg=OUT/'export_nbg_unify/network_binary.nb';base_nbg=SRC/'export_nbg_unify/network_binary.nb'
 status.update(phase='compiled',binary_bytes=nbg.stat().st_size,binary_sha256=sha(nbg),baseline_binary_sha256=sha(base_nbg),fc_weights_unchanged=records,total_files_bytes=sum(x.stat().st_size for x in OUT.rglob('*') if x.is_file()),input_order=json.loads((OUT/'conversion.json').read_text())['input_order'],output_order=json.loads((OUT/'conversion.json').read_text())['output_order'])
else:status['phase']='failed'
(ROOT/'prefill8-build-status.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps({k:v for k,v in status.items() if k not in ['command','fc_weights_unchanged']},indent=2),flush=True)
raise SystemExit(r.returncode)
