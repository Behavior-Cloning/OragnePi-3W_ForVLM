"""Offline error attribution using captured NPU x/y and actual exported weights.
FP32 matrix math is diagnostic; it does not reproduce NPU accumulation or timing.
"""
from pathlib import Path
import json,hashlib,re,time,traceback
import numpy as np,torch,yaml
T=Path(__file__).resolve().parents[1];D=T/'graphs/real-down-channel-balance';D.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
st={'state':'running','started':time.time()}
def save(): (D/'status.json').write_text(json.dumps(st,indent=2)+'\n')
save()
try:
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 C=T/'results/down-capture-c1280-replay-r1';cfg=json.loads((C/'config.json').read_text());ev={x['case']:x for x in map(json.loads,(C/'events.jsonl').read_text().splitlines()) if x['event']=='result'}
 S=T/'graphs/down-capture-c1280-m16-l2';Q=T/'graphs/candidates-down-i16-c1280-range16/c1280-m16-l2';q=yaml.safe_load((Q/'decoder.quantize').read_text())['quantize_parameters']
 def weight(folder,b):
  code=(folder/'export/vnn_decoder.c').read_text();label=f'@b{b}_fullconnect_116:weight';i=code.index('/* '+label+' */');j=code.index('/*',i+3);body=code[i:j]
  m=re.search(r'NEW_CONST_TENSOR\(const_tensor\[\d+\], attr, VSI_NN_TYPE_(FLOAT16|INT16), (\d+), (\d+)\)',body);assert m,body
  typ,off,n=m.groups();off,n=int(off),int(n);assert n==4864*896*2
  raw=(folder/'export/decoder.export.data').read_bytes()[off:off+n];assert len(raw)==n
  a=np.frombuffer(raw,dtype='<f2' if typ=='FLOAT16' else '<i2').reshape(896,4864).T.astype('float32').copy();fl=0
  if typ=='INT16':fl=int(re.search(r'attr.dtype.fl = (\d+);',body).group(1));a/=2**fl
  return torch.from_numpy(a),{'type':typ,'fl':fl,'offset':off,'bytes':n,'raw_sha256':hashlib.sha256(raw).hexdigest()}
 values=[[],[]];observed=[[],[]];rows=[]
 for f in cfg['fixtures']:
  name=f['key']+'-capture';path=C/(name+'.f32');assert sha(path)==ev[name]['output_sha256'];a=np.fromfile(path,'<f4');assert len(a)==16*(896+4*128)+32*4864+32*896 and np.isfinite(a).all()
  off=16*(896+4*128)
  for b in range(2):values[b].append(a[off+b*16*4864:off+(b+1)*16*4864].reshape(16,4864)[:f['valid_rows']].copy())
  off+=32*4864
  for b in range(2):observed[b].append(a[off+b*16*896:off+(b+1)*16*896].reshape(16,896)[:f['valid_rows']].copy())
  case=f['case'];static=64 if case.startswith('s64-') else 128;assert case.startswith(('s64-','s128-'))
  for i in range(f['valid_rows']):
   pos=f['position']+i;rows.append({'case':case,'position':pos,'region':'first_token' if pos==0 else 'first_chunk_other' if pos<16 else 'later_system' if pos<static else 'dynamic_and_template'})
 assert len(rows)==528
 results=[]
 with torch.inference_mode():
  for b in range(2):
   x=torch.from_numpy(np.concatenate(values[b]));npu=torch.from_numpy(np.concatenate(observed[b]));w,wm=weight(S,b)
   train=torch.tensor([r['region']=='dynamic_and_template' and r['case'].startswith('s64-') for r in rows]);held=torch.tensor([r['region']=='dynamic_and_template' and r['case'].startswith('s128-') for r in rows]);assert train.any() and held.any()
   xa=x[train].abs().amax(0).clamp_min(2**-20);wa=w.abs().amax(1).clamp_min(2**-20);variants=[];ref=x@w
   for alpha in [0.,.25,.5,.75,1.]:
    # x is divided by scale, rows of weight multiplied by scale.
    scale=2**((alpha*torch.log2(xa)-(1-alpha)*torch.log2(wa)).round().clamp(-8,8));scale/=2**torch.median(torch.log2(scale)).item()
    xx=x/scale;ww=w*scale[:,None];af=int(np.floor(np.log2(32767/float(xx[train].abs().max()))));wf=int(np.floor(np.log2(32767/float(ww.abs().max()))))
    qx=(xx*2**af).round().clamp(-32768,32767)/2**af;qw=(ww*2**wf).round().clamp(-32768,32767)/2**wf;out=qx@qw
    # FP16 input/output boundaries still require actual compiler/NPU validation.
    cases=[]
    for split,mask in [('calibration',train),('heldout_prompt',held)]:
     codes=(xx[mask]*2**af).round();diff=out[mask]-ref[mask];cases.append({'split':split,'rows':int(mask.sum()),'relative_l2_vs_fp32':float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(ref[mask])),'relative_l2_vs_observed_npu':float(torch.linalg.vector_norm(out[mask]-npu[mask])/torch.linalg.vector_norm(npu[mask])),'clipped_elements':int(((codes>32767)|(codes<-32768)).sum())})
    variants.append({'alpha':alpha,'activation_fl':af,'weight_fl':wf,'scales_pow2':[int(v) for v in torch.log2(scale).tolist()],'scale_min':float(scale.min()),'scale_max':float(scale.max()),'metrics':cases})
   results.append({'layer':b+2,'variants':variants})
 out={'scope':'Offline deterministic power-of-two channel balancing, actual original-NPU x/y, actual FP16 exported weights, FP32 matmul and simulated I16 grids. Calibration only s64 dynamic rows, heldout s128 dynamic rows from two previously reused prompts; not independent semantic validation. No prefix rows used to select scales. No NPU timing/implementation claim.','layers':results,'source_sha256':sha(Path(__file__)),'capture_config_sha256':sha(C/'config.json')}
 (D/'ANALYSIS.json').write_text(json.dumps(out,indent=2)+'\n');st.update(state='complete',ended=time.time(),analysis_sha256=sha(D/'ANALYSIS.json'));save()
except BaseException:st.update(state='failed',error=traceback.format_exc(),ended=time.time());save();raise
