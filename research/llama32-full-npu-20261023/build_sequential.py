"""Sequential 16 decoder + 4 full-vocabulary heads. Requires explicit execute."""
from model import *
import argparse,subprocess,shutil,time,os
p=argparse.ArgumentParser();p.add_argument('--precision',choices=['fp16','attn-fp16-mlp-w8'],required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
if not a.execute:print('Dry run: own-layer real calibration, 20 graphs sequential, preserve sources, cleanup only verified scratch.');raise SystemExit()
PYTHON='/opt/opi-lab/export-venv/bin/python';BUILD.mkdir(parents=True,exist_ok=True);final=BUILD/a.precision;final.mkdir(exist_ok=True);evidence=EVIDENCE/a.precision;evidence.mkdir(parents=True,exist_ok=True)
def run(label,cmd,timeout=1200):
 t=time.monotonic()
 with (evidence/(label+'.log')).open('w') as f:rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=timeout).returncode
 record=dict(label=label,command=cmd,seconds=time.monotonic()-t,returncode=rc)
 with (evidence/'stages.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
 if rc:raise RuntimeError(label+' failed')
if not (BUILD/'calibration-manifest.json').exists():run('calibration',[PYTHON,str(ROOT/'prepare_calibration.py'),'--execute'],1800)
rows=[]
for i in range(20):
 target=final/f'layer{i}.nb';manifest=evidence/f'layer{i}.json'
 if target.exists() and manifest.exists():
  old=json.loads(manifest.read_text());assert sha(target)==old['nbg_sha256'];rows.append(old);continue
 if shutil.disk_usage('/mnt/f').free<9*1024**3:raise RuntimeError('F reserve8GiB plus1GiB scratch guard')
 stage=BUILD/'scratch'/f'layer{i}';stage.mkdir(parents=True,exist_ok=True)
 run(f'{i}-export',[PYTHON,str(ROOT/'export_one.py'),'--index',str(i),'--stage',str(stage)],600)
 mixed=i<16 and a.precision=='attn-fp16-mlp-w8';base=stage/'compiled';selected=stage/'selected' if mixed else base
 run(f'{i}-import',[PYTHON,str(ROOT/'import_calibrate.py'),'--source',str(stage),'--output',str(base),'--layer',str(i),'--dtype','int16' if mixed else 'fp16','--timeout','300'])
 if mixed:run(f'{i}-compile',[PYTHON,str(ROOT/'compile_attention_fp16_mlp_w8.py'),'--base',str(base),'--output',str(selected)],600)
 else:run(f'{i}-compile',[PYTHON,str(ROOT/'compile_ovx.py'),'--directory',str(base)],600)
 nbg=selected/'export_nbg_unify/network_binary.nb';assert nbg.exists() and nbg.stat().st_size>1000
 meta=json.loads((stage/'export-manifest.json').read_text());meta.update(precision=a.precision if i<16 else 'fp16-full-vocabulary-head',nbg_sha256=sha(nbg),nbg_bytes=nbg.stat().st_size,source_sha256=sha(MODEL))
 # Preserve reviewable source/quantization/logs before removing regenerable scratch.
 ev=evidence/f'layer{i}';ev.mkdir(exist_ok=True)
 for name in ['decoder.json','decoder.quantize','conversion.json','export-result.json','export.log']:
  pth=selected/name
  if pth.exists():shutil.copyfile(pth,ev/name)
 c=selected/'export/vnn_decoder.c'
 if c.exists():shutil.copyfile(c,ev/'generated.c')
 tmp=target.with_suffix('.nb.partial');shutil.copyfile(nbg,tmp);assert sha(tmp)==meta['nbg_sha256'];os.replace(tmp,target);manifest.write_text(json.dumps(meta,indent=2));rows.append(meta)
 clean=dict(layer=i,scratch=str(stage),preserved_nbg=str(target),sha256=meta['nbg_sha256'])
 (ev/'cleanup.json').write_text(json.dumps(clean,indent=2));shutil.rmtree(stage)
 print('complete graph',i,'bytes',meta['nbg_bytes'],flush=True)
result=dict(source=str(MODEL),source_precision=MODEL.name+' dequantized; not BF16',capacity=json.loads((BUILD/'inputs.json').read_text())['capacity'],precision=a.precision,decoder_layers=16,head_shards=4,vocab=128256,graphs=rows,npu_executed=False,frontend_cpu=['embedding lookup','RoPE sin/cos construction','KV copy','dispatch','greedy token selection'],npu_ops=['all decoder projections/norm/rotary multiplication/attention/MLP','final RMSNorm','full vocabulary head'],prefill='scalar NPU; no CPU prefill and no duplicate batch graph')
(final/'manifest.json').write_text(json.dumps(result,indent=2));(evidence/'manifest.json').write_text(json.dumps(result,indent=2));print('ALL20_GRAPHS_COMPILED',flush=True)
