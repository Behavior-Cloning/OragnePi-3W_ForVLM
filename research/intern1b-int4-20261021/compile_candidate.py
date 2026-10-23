"""Isolated per-channel W4 candidate from the established W8 Intern1B graph."""
from pathlib import Path
import argparse,json,hashlib,subprocess,time,os,shutil
import numpy as np,yaml
ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['decode','prefill','head'],required=True);ap.add_argument('--layer',type=int,required=True);a=ap.parse_args()
root=Path('/opt/opi-lab/intern1b-int4-20261021');src=Path('/opt/opi-lab/source-validation-20261017/gguf-fresh/export')
variant={'decode':'w8-fused2-c1280-cap1-headmse','prefill':'w8-fused2-c1280-t16-cap1-productwide','head':'w8-balanced'}[a.kind]
base=src/variant/('l'+str(a.layer));out=root/a.kind/('l'+str(a.layer));out.mkdir(parents=True,exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
records={n:sha(base/n) for n in ['decoder.json','decoder.data','decoder.quantize','tensor-inputmeta.yml']}
for n in ['decoder.json','decoder.data','tensor-inputmeta.yml']:shutil.copyfile(base/n,out/n)
q=yaml.safe_load((base/'decoder.quantize').read_text());data=np.load(base/'decoder.data',allow_pickle=True).item();changed=[];retained=[]
for n,d in data.items():
 key='@'+n+':weight'
 if 'fullconnect' not in n or 'weight' not in d or key not in q['quantize_parameters']:continue
 before=q['quantize_parameters'][key]
 if before.get('qtype') not in ['i8','int8']:
  retained.append({'name':n,'dtype':before.get('qtype'),'shape':list(d['weight'].shape)});continue
 w=d['weight'];scale=np.maximum(np.abs(w).max(axis=0)/7,np.finfo(np.float32).tiny)
 q['quantize_parameters'][key]={'qtype':'i4','quantizer':'perchannel_symmetric_affine','rounding':'rtne','scale':scale.tolist(),'channel_dim':-1}
 reconstructed=np.clip(np.rint(w/scale),-7,7)*scale
 changed.append({'name':n,'shape':list(w.shape),'source_type':before.get('qtype'),'target_type':'i4','relative_l2':float(np.linalg.norm(w-reconstructed)/np.linalg.norm(w))})
assert changed
(out/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False))
manifest={'kind':a.kind,'layer':a.layer,'source':str(base),'source_hashes':records,'changed':changed,'retained':retained,'note':'Requested native i4 in ACUITY; hardware execution and precision unverified. Activations/KV and FP16 exceptions retained from W8; embeddings remain FP32.','script_sha256':sha(Path(__file__))}
cmd=['docker','run','--rm','--name','intern-int4-'+a.kind+'-'+str(a.layer),'--network','none','--cpus','4','--memory','8g','-e','TF_CPP_MIN_LOG_LEVEL=3','-e','OMP_NUM_THREADS=4','-e','VSIMULATOR_SHADER_CORE_COUNT=1','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B','-v',str(root)+':/work','-w','/work/'+a.kind+'/l'+str(a.layer),'ubuntu-npu:v2.0.10.2','python3','/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py','export','ovxlib','--model','decoder.json','--model-data','decoder.data','--model-quantize','decoder.quantize','--dtype','quantized','--with-input-meta','tensor-inputmeta.yml','--target-ide-project','linux64','--pack-nbg-unify','--optimize','VIP9000NANODI_PLUS_PID0X1000003B','--viv-sdk','/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools','--output-path','export/decoder']
manifest['command']=cmd;t=time.monotonic()
with (out/'export.log').open('w') as log:
 try:z=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300);manifest['exit']=z.returncode
 except subprocess.TimeoutExpired:
  subprocess.run(['docker','stop','-t','2','intern-int4-'+a.kind+'-'+str(a.layer)],stdout=log,stderr=subprocess.STDOUT);manifest['exit']=124
manifest['compile_seconds']=time.monotonic()-t
nb=out/'export_nbg_unify/network_binary.nb'
if nb.exists():manifest.update(binary_bytes=nb.stat().st_size,binary_sha256=sha(nb))
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items() if k not in ['command','changed','source_hashes']},indent=2),flush=True)
raise SystemExit(manifest['exit'])
