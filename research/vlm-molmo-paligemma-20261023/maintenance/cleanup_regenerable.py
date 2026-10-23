from pathlib import Path
import hashlib,json,time,os
r=Path('/mnt/f/Files/OPi-Lab/repos/OragnePi-3W_ForVLM');out=r/'research/vlm-molmo-paligemma-20261023/maintenance/cleanup-manifest.jsonl'
candidates=[]
for model in ['laya','nano33']:
 d=r/'research/jev-validation/models'/model
 assert (d/'model.safetensors').is_file()
 assert (r/('research/jev-validation/export_'+model.replace('nano33','nano')+'_onnx.py')).is_file()
 candidates.append((d/(model+'-fp32.onnx'),'direct_F','Regenerable FP32 ONNX; original safetensors, exporter, INT8 deployment retained'))
b=Path('/opt/opi-lab/intern1b-int4-20261021')
for family in ['decode','prefill','head']:
 for d in (b/family).glob('l*'):
  if not d.is_dir():continue
  if not list(d.rglob('network_binary.nb')):continue
  for f in [d/'decoder.data',d/'export/decoder.export.data']:
   if f.is_file():candidates.append((f,'WSL_internal','Regenerable compiler weight intermediate; final NBG, quantization, logs, source retained'))
b=Path('/opt/opi-lab/minicpm-v46-npu-20261023')
for f in b.rglob('decoder.export.data'):
 if f.parent.name!='export':continue
 if not list(f.parent.parent.rglob('network_binary.nb')):continue
 candidates.append((f,'WSL_internal','Regenerable compiler export intermediate; final NBG and source retained'))
summary={};start=time.time()
with out.open('a') as log:
 for f,storage,reason in candidates:
  if not f.is_file():continue
  size=f.stat().st_size;h=hashlib.sha256()
  with f.open('rb') as inp:
   for chunk in iter(lambda:inp.read(8*1024*1024),b''):h.update(chunk)
  record={'path':str(f),'bytes':size,'sha256':h.hexdigest(),'storage':storage,'reason':reason,'state':'verified_before_delete','time':time.time()}
  log.write(json.dumps(record)+'\n');log.flush();os.fsync(log.fileno())
  f.unlink();record['state']='deleted';log.write(json.dumps(record)+'\n');log.flush()
  summary[storage]=summary.get(storage,0)+size
print(json.dumps({'deleted_bytes':summary,'files':len(candidates),'seconds':time.time()-start}),flush=True)
