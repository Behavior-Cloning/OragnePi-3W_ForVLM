"""Offline error attribution using captured NPU x/y and actual exported weights.
FP32 matrix math is diagnostic; it does not reproduce NPU accumulation or timing.
"""
from pathlib import Path
import json,hashlib,re,time,traceback
import numpy as np,torch,yaml
T=Path(__file__).resolve().parents[1];D=T/'graphs/real-down-power2-row-scale';D.mkdir(exist_ok=False)
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
   x=torch.from_numpy(np.concatenate(values[b]));npu=torch.from_numpy(np.concatenate(observed[b]));w,wm=weight(S,b);wq,wqm=weight(Q,b);ref=x@w;rr=[]
   # Power-of-two scales avoid the additional FP16 rounding from arbitrary division/restoration.
   peak=x.abs().amax(1,keepdim=True).clamp_min(2**-14)
   floor_scale=torch.pow(2.0,torch.floor(torch.log2(peak))).clamp(2**-14,2**15)
   ceil_scale=torch.pow(2.0,torch.ceil(torch.log2(peak))).clamp(2**-14,2**15)
   variants=[('max-fl14',peak,14),('floor-pow2-fl14',floor_scale,14),('ceil-pow2-fl14',ceil_scale,14),('ceil-pow2-fl15',ceil_scale,15)]
   for name,scale,fl in variants:
    scaled=(x/scale).half().float();raw=(scaled*2**fl).round();qx=raw.clamp(-32768,32767)/2**fl
    inner=qx@wq
    for boundary in ['fp32_ideal','fp16_boundaries']:
     pred=inner*scale if boundary=='fp32_ideal' else (inner.half().float()*scale).half().float()
     for region in ['first_token','first_chunk_other','later_system','dynamic_and_template']:
      mask=torch.tensor([r['region']==region for r in rows]);v=pred[mask]
      rr.append({'region':region,'variant':name,'activation_fl':fl,'boundary':boundary,'rows':int(mask.sum()),'clipped_elements':int(((raw[mask]<-32768)|(raw[mask]>32767)).sum()),'relative_l2_vs_fp32':float(torch.linalg.vector_norm(v-ref[mask])/torch.linalg.vector_norm(ref[mask])),'relative_l2_vs_observed_npu':float(torch.linalg.vector_norm(v-npu[mask])/torch.linalg.vector_norm(npu[mask])),'max_abs_vs_npu':float((v-npu[mask]).abs().max()),'finite':bool(torch.isfinite(v).all())})
   results.append({'layer':b+2,'metrics':rr})
 out={'scope':'Offline per-row dynamic normalization using actual original-NPU x, actual exported FP16/I16 weights. No calibration fit. FP32 reduction/division/matmul plus specified FP16 rounding; no NPU accumulation or compiled kernels, no speed/full-model/semantic quality claim. Two reused prompts only.','layers':results,'source_sha256':sha(Path(__file__)),'capture_config_sha256':sha(C/'config.json')}
 (D/'ANALYSIS.json').write_text(json.dumps(out,indent=2)+'\n');st.update(state='complete',ended=time.time(),analysis_sha256=sha(D/'ANALYSIS.json'));save()
except BaseException:st.update(state='failed',error=traceback.format_exc(),ended=time.time());save();raise
