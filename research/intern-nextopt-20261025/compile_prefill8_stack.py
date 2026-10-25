"""Expand only query shape8; freeze baseline graph data, quantization and constants."""
from pathlib import Path
import os,json,subprocess,time,hashlib,yaml
ROOT=Path(__file__).resolve().parent;BASE=Path('/opt/opi-lab/source-validation-20261017/gguf-fresh/export')
SRC=BASE/'w8-fused2-c1280-t16-cap1-productwide';BUILD=Path('/opt/opi-lab/intern-nextopt-20261025/prefill8')
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def freef():s=os.statvfs('/mnt/f');return s.f_bavail*s.f_frsize
records=[];started=time.time();status=ROOT/'prefill8-stack-status.json'
def save(state):status.write_text(json.dumps({'phase':state,'pid':os.getpid(),'started_unix':started,'f_free_bytes':freef(),'blocks':records},indent=2)+'\n')
for first in range(0,24,2):
 assert freef()>8*1024**3,'F free below reserve'
 old=SRC/f'l{first}';new=BUILD/f'l{first}';conv=json.loads((old/'conversion.json').read_text())
 variants={Path(x['folder']).parts[-2] for x in conv['sources']};assert len(variants)==1
 variant=variants.pop();assert conv['capacity']==1280 and conv['query_tokens']==16
 for flag in ['compact_gqa','split_kv_attention','split_softmax_i16','split_current_matmul']:assert conv[flag]
 for flag in ['attention_fp16','fp16_kv_cache','key_cache_transposed','native_rmsnorm','value_output_fp16','split_probability_i16']:assert not conv[flag]
 assert conv['fc_weight_dfp']=='gateup' and conv['fc_weight_dfp_method']=='mse' and conv['fc_weight_dfp_max_extra_bits']==1
 cmd=['/opt/opi-lab/export-venv/bin/python',str(ROOT/'compiler-source/compile_fused_decoder.py'),'--source',str(BASE),'--variant',variant,'--results-root','/opt/opi-lab','--first',str(first),'--width','2','--output',str(new),'--timeout','300','--compact-gqa','--capacity','1280','--tokens','8','--fc-weight-dfp','gateup','--fc-weight-dfp-method','mse','--fc-weight-dfp-max-extra-bits','1','--split-kv-attention','--split-softmax-i16','--split-current-matmul']
 for key,arg in [('mlp_product_headroom','--mlp-product-headroom'),('query_headroom','--query-headroom')]:
  for layer,bits in conv[key]:cmd += [arg,str(layer),str(bits)]
 row={'first':first,'source':str(old),'output':str(new),'command':cmd,'baseline_binary_sha256':sha(old/'export_nbg_unify/network_binary.nb')};records.append(row);save('compiling')
 t=time.monotonic()
 if first==0:
  assert (new/'export_nbg_unify/network_binary.nb').exists();row['reused_existing_first_block']=True
 else:
  assert not new.exists(),'do not overwrite previous graph'
  with (ROOT/f'prefill8-l{first}-compile.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=360)
  row['returncode']=r.returncode
  if r.returncode:save('failed');raise SystemExit(r.returncode)
 row['seconds']=time.monotonic()-t
 assert yaml.safe_load((old/'decoder.quantize').read_text())==yaml.safe_load((new/'decoder.quantize').read_text()),'quantization changed'
 row['all_quantize_parameters_identical']=True;row['exact_constants']=[]
 # Both original FP32 graph data and actual exported quantized byte payload must match.
 # After verification reuse baseline payload via symlink, avoiding duplicated scratch.
 for name in ['decoder.data','export/decoder.export.data']:
  a,b=old/name,new/name;ha,hb=sha(a),sha(b);assert ha==hb,(first,name,'constant bytes changed')
  row['exact_constants'].append({'file':name,'bytes':b.stat().st_size,'sha256':hb,'baseline':str(a),'replaced_by_verified_symlink':True})
  if not b.is_symlink():b.unlink();b.symlink_to(a)
 nb=new/'export_nbg_unify/network_binary.nb';row.update(binary_bytes=nb.stat().st_size,binary_sha256=sha(nb),phase='compiled')
 save('compiling');print('COMPILED',first,row['binary_bytes'],row['binary_sha256'],flush=True)
save('complete')
manifest={'capacity':1280,'tokens':8,'fusion_width':2,'blocks':[{k:r[k] for k in ['first','output','binary_bytes','binary_sha256','baseline_binary_sha256','all_quantize_parameters_identical','exact_constants']} for r in records],
 'all_binary_bytes':sum(r['binary_bytes'] for r in records),'elapsed_seconds':time.time()-started,'scope':'Shape-only t8 overlay; all actual constant payloads and quantizers equal each t16 baseline; full-stack numeric test still required.'}
(ROOT/'prefill8-stack-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('ALL_12_T8_BLOCKS_COMPLETE',flush=True)
