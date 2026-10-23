from pathlib import Path
import numpy as np,yaml,json,subprocess,os,time,sys,hashlib,math,copy
r=Path(__file__).resolve().parents[1];b=Path('/opt/opi-lab/minicpm-v46-npu-20261023');name=sys.argv[1];variant=sys.argv[2] if len(sys.argv)>2 else 'all'
src=(b/'w8scale-imported'/name) if (b/'w8scale-imported'/name/'decoder.data').exists() else b/'w8scale-fp16'/name;o=b/('w8scaled-'+variant)/name;o.mkdir(parents=True,exist_ok=True)
for f in ['decoder.data','tensor-inputmeta.yml']:
 if not (o/f).exists():os.link(src/f,o/f)
g=json.loads((src/'decoder.json').read_text());q=yaml.safe_load((src/'decoder.quantize').read_text());original=copy.deepcopy(q['quantize_parameters']);qp=q['quantize_parameters'];data=np.load(src/'decoder.data',allow_pickle=True).item();cal=json.loads((r/'results/w8-household-calibration.json').read_text());hashes={v:k for k,v in cal['weight_sha256'].items()};changes=[]
for k in qp:
 if not k.endswith((':weight',':bias')):qp[k]={'qtype':'float16','quantizer':'float16'}
def quant(maxval):
 return {'qtype':'i16','quantizer':'dynamic_fixed_point','rounding':'rtne','fl':int(math.floor(math.log2(32767/max(maxval*2,1e-9)))),'min_value':-maxval*2,'max_value':maxval*2}
for i,(n,v) in enumerate(list(g['Layers'].items())):
 if v['op']!='fullconnect':continue
 qp.setdefault('@'+n+':bias',{'qtype':'i64','quantizer':'dynamic_fixed_point','rounding':'rtne','fl':24});w=data[n]['weight'];key=name if name.startswith('head') else hashes[hashlib.sha256(w.tobytes()).hexdigest()];ranges=cal['ranges'][key]
 if variant=='mlponly' and '.linear_attn.' in key or variant=='protect' and key.endswith(('in_proj_a','in_proj_b')):
  qp['@'+n+':weight']={'qtype':'float16','quantizer':'float16'};qp.pop('@'+n+':bias',None);changes.append({'name':n,'module':key,'weight':'float16'});continue
 scale=np.maximum(abs(w).max(axis=0)/127,np.finfo(np.float32).tiny);qp['@'+n+':weight']={'qtype':'i8','quantizer':'perchannel_symmetric_affine','rounding':'rtne','scale':scale.tolist(),'channel_dim':-1}
 before=v['inputs'][0];conv='fc_input_convert_'+str(20000+i)
 g['Layers'][conv]={'name':conv,'op':'dtype_converter','parameters':{},'inputs':[before],'outputs':['out0']}
 qi=quant(ranges['input_absmax']);qo=quant(ranges['output_absmax']);qp['@'+conv+':out0']=qi;qp['@'+n+':out0']=qo;v['inputs'][0]='@'+conv+':out0';changes.append({'name':n,'module':key,'weight':'int8','input_quant':qi,'output_quant':qo})
(o/'decoder.json').write_text(json.dumps(g,indent=2));(o/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False));(o/'changes.json').write_text(json.dumps(changes,indent=2))
container='minicpm46-cal-'+name+'-'+variant
cmd=['docker','run','--rm','--name',container,'--network','none','--cpus','4','--memory','8g','-e','TF_CPP_MIN_LOG_LEVEL=3','-e','OMP_NUM_THREADS=4','-e','VSIMULATOR_SHADER_CORE_COUNT=1','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B','-v',str(b)+':/work','-w','/work/'+str(o.relative_to(b)),'ubuntu-npu:v2.0.10.2','python3','/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py','export','ovxlib','--model','decoder.json','--model-data','decoder.data','--model-quantize','decoder.quantize','--dtype','quantized','--with-input-meta','tensor-inputmeta.yml','--target-ide-project','linux64','--pack-nbg-unify','--optimize','VIP9000NANODI_PLUS_PID0X1000003B','--viv-sdk','/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools','--output-path','export/decoder']
t=time.time()
with (o/'export.log').open('w') as f:
 try:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=180);code=z.returncode
 except subprocess.TimeoutExpired:subprocess.run(['docker','stop','-t','2',container]);code=124
row={'name':name,'variant':variant,'exit':code,'seconds':time.time()-t,'binaries':[{'path':str(p),'bytes':p.stat().st_size} for p in o.rglob('network_binary.nb')]};(o/'conversion.json').write_text(json.dumps(row,indent=2));print(row,flush=True)

raise SystemExit(code)
