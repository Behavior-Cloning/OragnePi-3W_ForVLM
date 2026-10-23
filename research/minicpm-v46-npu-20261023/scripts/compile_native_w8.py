from pathlib import Path
import numpy as np,yaml,json,subprocess,os,time,sys,copy,shutil
r=Path(__file__).resolve().parents[1];b=Path('/opt/opi-lab/minicpm-v46-npu-20261023');name=sys.argv[1];variant=sys.argv[2] if len(sys.argv)>2 else 'gate'
src=b/'w8cal-all'/name;o=b/('w8native-'+variant)/name;o.mkdir(parents=True,exist_ok=True);g=json.loads((src/'decoder.json').read_text());q=yaml.safe_load((src/'decoder.quantize').read_text());nodes=g['Layers'];qp=q['quantize_parameters'];data=np.load(src/'decoder.data',allow_pickle=True).item();changes=[]
def key(v):return v[1:].rsplit(':',1)[0]
for n,v in list(nodes.items()):
 if v['op']!='multiply':continue
 for ref in v.get('inputs',[]):
  div=nodes[key(ref)]
  if div['op']!='real_div':continue
  d=nodes[key(div['inputs'][1])]
  while d['op'] in ['reshape','permute']:d=nodes[key(d['inputs'][0])]
  if d['op']!='clipbyvalue':continue
  eps=float(d['parameters']['clip_value_min'])**2
  if abs(eps-1e-6)>1e-10:continue
  source=div['inputs'][0];source_op=nodes[key(source)]['op'];gate=source_op=='matmul'
  if variant=='gate' and not gate:continue
  width=128 if gate else 1024
  assert qp['@'+n+':out0']['qtype']=='float16'
  v.update(op='rmsnormalize',parameters={'axis':-1,'eps':eps},inputs=[source]);data[n]={'scale':np.ones(width,np.float32)};qp['@'+n+':scale']={'qtype':'float16','quantizer':'float16'};changes.append({'node':n,'input':source,'width':width,'eps':eps});break
assert len(changes)==(1 if variant=='gate' else 3),changes
live=set()
def keep(n):
 if n in live:return
 live.add(n)
 for ref in nodes[n].get('inputs',[]):keep(key(ref))
for n,v in nodes.items():
 if v['op']=='output':keep(n)
g['Layers']={n:v for n,v in nodes.items() if n in live};q['quantize_parameters']={n:v for n,v in qp.items() if key(n) in live};data={n:v for n,v in data.items() if n in live}
with (o/'decoder.data').open('wb') as f:np.save(f,data)
(o/'decoder.json').write_text(json.dumps(g,indent=2));(o/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False));(o/'changes.json').write_text(json.dumps(changes,indent=2));shutil.copyfile(src/'tensor-inputmeta.yml',o/'tensor-inputmeta.yml')
container='minicpm46-native-'+name+'-'+variant
cmd=['docker','run','--rm','--name',container,'--network','none','--cpus','2','--memory','5g','-e','TF_CPP_MIN_LOG_LEVEL=3','-e','OMP_NUM_THREADS=2','-e','VSIMULATOR_SHADER_CORE_COUNT=1','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B','-v',str(b)+':/work','-w','/work/'+str(o.relative_to(b)),'ubuntu-npu:v2.0.10.2','python3','/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py','export','ovxlib','--model','decoder.json','--model-data','decoder.data','--model-quantize','decoder.quantize','--dtype','quantized','--with-input-meta','tensor-inputmeta.yml','--target-ide-project','linux64','--pack-nbg-unify','--optimize','VIP9000NANODI_PLUS_PID0X1000003B','--viv-sdk','/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools','--output-path','export/decoder']
t=time.time()
with (o/'export.log').open('w') as f:
 try:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=180);code=z.returncode
 except subprocess.TimeoutExpired:subprocess.run(['docker','stop','-t','2',container]);code=124
row={'name':name,'variant':variant,'exit':code,'seconds':time.time()-t,'changes':changes,'binaries':[{'path':str(p),'bytes':p.stat().st_size} for p in o.rglob('network_binary.nb')]};(o/'conversion.json').write_text(json.dumps(row,indent=2));print(row,flush=True)

raise SystemExit(code)
