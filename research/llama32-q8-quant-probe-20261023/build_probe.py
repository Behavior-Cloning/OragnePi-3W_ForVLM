"""Sequential bounded probe compilation. This script never accesses the board."""
from model import *
import subprocess,time,shutil,os
PY='/opt/opi-lab/export-venv/bin/python';EV=ROOT/'evidence';EV.mkdir(exist_ok=True);FINAL=BUILD/'nbg';FINAL.mkdir(exist_ok=True)
def run(label,cmd):
 assert shutil.disk_usage('/mnt/f').free > 9*1024**3
 t=time.monotonic()
 with (EV/(label+'.log')).open('w') as f:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=1000)
 with (EV/'stages.jsonl').open('a') as f:f.write(json.dumps(dict(label=label,command=cmd,seconds=time.monotonic()-t,returncode=z.returncode))+'\n')
 if z.returncode:raise RuntimeError(label+' failed')
 print(label,'completed',flush=True)
def retain(selected,name):
 nbg=selected/'export_nbg_unify/network_binary.nb';assert nbg.exists();dest=FINAL/(name+'.nb');shutil.copyfile(nbg,dest);digest=sha(nbg);assert sha(dest)==digest
 ev=EV/name;ev.mkdir(exist_ok=True)
 for fn in ['decoder.json','decoder.quantize','conversion.json','export.log']:
  if (selected/fn).exists():shutil.copyfile(selected/fn,ev/fn)
 c=selected/'export/vnn_decoder.c'
 if c.exists():shutil.copyfile(c,ev/'generated.c')
 row=dict(name=name,path=str(dest),bytes=dest.stat().st_size,sha256=digest,source_sha256=sha(MODEL));(EV/(name+'.json')).write_text(json.dumps(row,indent=2));print('READY',json.dumps(row),flush=True)
stage=BUILD/'scratch-layer0';base=stage/'compiled';stage.mkdir(exist_ok=True)
run('layer0-export',[PY,str(ROOT/'export_one.py'),'--index','0','--stage',str(stage)])
run('layer0-calibrate',[PY,str(ROOT/'import_calibrate.py'),'--source',str(stage),'--output',str(base),'--dtype','int16','--timeout','300'])
run('layer0-w8',[PY,str(ROOT/'compile_w8.py'),'--base',str(base),'--output',str(stage/'allw8'),'--float-sensitive','--float-residual','--timeout','300'])
retain(stage/'allw8','layer0-allw8')
run('layer0-mixed',[PY,str(ROOT/'compile_attention_fp16_mlp_w8.py'),'--base',str(base),'--output',str(stage/'mixed')])
retain(stage/'mixed','layer0-mixed')
shutil.copyfile(stage/'export-manifest.json',EV/'layer0-export-manifest.json')
# Keep calibration metadata; discard only this probe's regenerable bulky scratch.
for fn in ['decoder.json','decoder.quantize','conversion.json']:
 shutil.copyfile(base/fn,EV/('layer0-base-'+fn))
shutil.rmtree(stage)
print('DECODER_PROBES_READY',flush=True)
for idx in range(16,20):
 stage=BUILD/('scratch-head'+str(idx-16));base=stage/'compiled';stage.mkdir(exist_ok=True)
 run(f'head{idx-16}-export',[PY,str(ROOT/'export_one.py'),'--index',str(idx),'--stage',str(stage)])
 run(f'head{idx-16}-calibrate',[PY,str(ROOT/'import_calibrate.py'),'--source',str(stage),'--output',str(base),'--layer',str(idx),'--dtype','int16','--timeout','300'])
 run(f'head{idx-16}-w8',[PY,str(ROOT/'compile_w8.py'),'--base',str(base),'--output',str(stage/'allw8'),'--expected-linears','1','--float-head-norm','--timeout','300'])
 retain(stage/'allw8',f'head{idx-16}-w8');shutil.copyfile(stage/'export-manifest.json',EV/(f'head{idx-16}-export-manifest.json'));shutil.rmtree(stage)
print('ALL6_PROBES_COMPILED',flush=True)
