"""All attention/IO floating, three MLP FC W8 with explicit calibrated A16 boundaries."""
from pathlib import Path
import argparse,copy,hashlib,json,os,shutil,subprocess,time,yaml
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=Path('/opt/opi-lab/llama32-compat-20261023/compiled'));p.add_argument('--output',type=Path,default=Path('/opt/opi-lab/llama32-compat-20261023/attention-fp16-mlp-w8'));a=p.parse_args()
r=Path(__file__).resolve().parent;root=Path('/opt/opi-lab');base=a.base.resolve();out=a.output.resolve();assert shutil.disk_usage('/mnt/f').free>8*1024**3
out.mkdir(exist_ok=True,parents=True)
for n in ('decoder.data','decoder.json','tensor-inputmeta.yml','dataset.txt'):
 if not (out/n).exists():os.link(base/n,out/n)
 assert os.path.samefile(base/n,out/n)
original=yaml.safe_load((base/'decoder.quantize').read_text());g=json.loads((base/'decoder.json').read_text())['Layers']
q=copy.deepcopy(original);qp=q['quantize_parameters'];oq=original['quantize_parameters']
for k in list(qp):
 if k.endswith(':bias'):qp.pop(k)
 else:qp[k]={'qtype':'float16','quantizer':'float16'}
# Find the residual that adds hidden, then trace only the MLP output branch.
h=next(k for k,v in g.items() if v.get('name')=='hidden');res=next(k for k,v in g.items() if v['op']=='add' and '@'+h+':out0' in v.get('inputs',[]));end=next(k for k,v in g.items() if v.get('name')=='hidden_out');seen=set();mlp=set()
def visit(n):
 if n==res or n in seen:return
 seen.add(n)
 if g[n]['op']=='fullconnect':mlp.add(n)
 for x in g[n].get('inputs',[]):
  if x.startswith('@'):visit(x[1:].split(':')[0])
visit(end);assert len(mlp)==3,mlp
restored=set()
def restore_output(n):
 key='@'+n+':out0';v=copy.deepcopy(oq[key]);assert v['qtype']=='i16';qp[key]=v;restored.add(n)
data=None
for n in sorted(mlp):
 key='@'+n+':weight'
 if oq[key]['qtype']=='i8':qp[key]=copy.deepcopy(oq[key])
 else:
  if data is None:data=np.load(base/'decoder.data',allow_pickle=True).item() # trusted own compiler-generated data
  w=data[n]['weight'];scale=np.maximum(np.max(np.abs(w),axis=0)/127,np.finfo(np.float32).tiny)
  qp[key]={'qtype':'i8','quantizer':'perchannel_symmetric_affine','rounding':'rtne','scale':scale.tolist(),'channel_dim':-1}
 bias='@'+n+':bias'
 if bias in oq:qp[bias]=copy.deepcopy(oq[bias])
 restore_output(n)
 # Layout-only nodes between FC and its actual source share the A16 range.
 assert len(g[n]['inputs'])==1
 x=g[n]['inputs'][0][1:].split(':')[0]
 while True:
  restore_output(x)
  if g[x]['op'] not in ('reshape','permute'):break
  assert len(g[x]['inputs'])==1;x=g[x]['inputs'][0][1:].split(':')[0]
 # Carry FC output precision through harmless reshapes; nonlinear consumer stays FP16.
 todo=[n]
 while todo:
  src=todo.pop()
  for target,v in g.items():
   if v['op'] in ('reshape','permute') and v.get('inputs')==['@'+src+':out0']:
    if target not in restored:restore_output(target);todo.append(target)
for n,v in g.items():
 if v['op']=='input' or v.get('name') in ('hidden_out','new_k','new_v') or v['op']=='softmax':
  key='@'+n+':out0';qp.setdefault(key,{'qtype':'float16','quantizer':'float16'});assert qp[key]['qtype']=='float16',(n,qp[key])
(out/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False))
name='llama32-attnfp16-mlpw8';relative=out.relative_to(root)
cmd=['docker','run','--rm','--name',name,'--network','none','--cpus','2','--memory','5g','-e','TF_CPP_MIN_LOG_LEVEL=3','-e','OMP_NUM_THREADS=2','-e','VSIMULATOR_SHADER_CORE_COUNT=1','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B','-v',str(root)+':/work','-w','/work/'+str(relative),'ubuntu-npu:v2.0.10.2','python3','/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py','export','ovxlib','--model','decoder.json','--model-data','decoder.data','--model-quantize','decoder.quantize','--dtype','quantized','--with-input-meta','tensor-inputmeta.yml','--target-ide-project','linux64','--pack-nbg-unify','--optimize','VIP9000NANODI_PLUS_PID0X1000003B','--viv-sdk','/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools','--output-path','export/decoder']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
t=time.monotonic();report={'scope':'attention/IO FP16, 3MLP weights W8, only FC-adjacent A16; fresh Chinese calibration specified by input base directory','source_sha256':{n:sha(base/n) for n in ('decoder.json','decoder.data','decoder.quantize')},'mlp_linears':sorted(mlp),'restored_i16_nodes':{n:g[n]['name'] for n in sorted(restored)},'command':cmd,'compiler_script_sha256':sha(Path(__file__))}
with (out/'export.log').open('w') as log:
 try:z=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300);code=z.returncode
 except subprocess.TimeoutExpired:
  subprocess.run(['docker','stop','-t','2',name],stdout=log,stderr=subprocess.STDOUT,timeout=15);code=124
report.update(exit_code=code,seconds=time.monotonic()-t);report['binaries']=[dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f)) for f in out.rglob('network_binary.nb')]
(out/'conversion.json').write_text(json.dumps(report,indent=2)+'\n');(r/'attention-fp16-mlp-w8-conversion.json').write_text(json.dumps(report,indent=2)+'\n');shutil.copyfile(out/'export.log',r/'attention-fp16-mlp-w8-compile.log');print(json.dumps(report,indent=2),flush=True)
raise SystemExit(code)
