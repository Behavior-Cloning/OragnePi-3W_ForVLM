"""Offline error attribution using captured NPU x/y and actual exported weights.
FP32 matrix math is diagnostic; it does not reproduce NPU accumulation or timing.
"""
from pathlib import Path
import json,hashlib,re,time,traceback
import numpy as np,torch,yaml
T=Path(__file__).resolve().parents[1];D=T/'graphs/real-down-error-budget';D.mkdir(exist_ok=False)
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
  source=np.load(S/'decoder.data',allow_pickle=True).item()
  for b in range(2):
   x=torch.from_numpy(np.concatenate(values[b]));npu=torch.from_numpy(np.concatenate(observed[b]));w,wm=weight(S,b);wq,wqm=weight(Q,b)
   source_half=torch.from_numpy(source[f'b{b}_fullconnect_116']['weight']).half().float();assert torch.equal(w,source_half);assert np.count_nonzero(source[f'b{b}_fullconnect_116']['bias'])==0
   fl=q[f'@down_i16_{2000+b}:out0']['fl'];ofl=q[f'@b{b}_fullconnect_116:out0']['fl'];xq=(x*2**fl).round().clamp(-32768,32767)/2**fl
   base=x@w;combined=xq@wq
   pred={'float32_exact_export_weights':base,'float16_output_rounding':base.half().float(),'activation_only':xq@w,'weight_only':x@wq,'activation_and_weight':combined,'with_output_grid':(combined*2**ofl).round().clamp(-32768,32767)/2**ofl,'ideal_input_residual_correction':combined+(x-xq)@wq,'ideal_weight_residual_correction':combined+xq@(w-wq)}
   rr=[]
   for region in ['first_token','first_chunk_other','later_system','dynamic_and_template']:
    mask=torch.tensor([r['region']==region for r in rows]);ref=base[mask];actual=npu[mask]
    for name,y in pred.items():
     v=y[mask];rr.append({'region':region,'rows':int(mask.sum()),'variant':name,'relative_l2_vs_fp32':float(torch.linalg.vector_norm(v-ref)/torch.linalg.vector_norm(ref).clamp_min(1e-30)),'relative_l2_vs_observed_npu':float(torch.linalg.vector_norm(v-actual)/torch.linalg.vector_norm(actual).clamp_min(1e-30)),'max_abs_vs_observed_npu':float((v-actual).abs().max())})
   results.append({'layer':b+2,'original_weight':wm,'candidate_weight':wqm,'activation_fl':fl,'output_fl':ofl,'weight_relative_l2':float(torch.linalg.vector_norm(wq-w)/torch.linalg.vector_norm(w)),'metrics':rr})
 out={'scope':'Offline FP32 attribution from34 real original-NPU capture windows (2 reused prompts),528valid rows. No candidate NPU propagation, no speed or semantic quality claim; hypothetical residual corrections omit extra branch quantization and NPU accumulation. Not independent holdout.','rows':rows,'layers':results,'sources':{str(f.relative_to(T)):sha(f) for f in [Path(__file__),C/'config.json',C/'events.jsonl',S/'export/vnn_decoder.c',S/'export/decoder.export.data',Q/'export/vnn_decoder.c',Q/'export/decoder.export.data',Q/'decoder.quantize']}}
 (D/'ANALYSIS.json').write_text(json.dumps(out,indent=2)+'\n');st.update(state='complete',ended=time.time(),analysis_sha256=sha(D/'ANALYSIS.json'));save()
except BaseException:st.update(state='failed',error=traceback.format_exc(),ended=time.time());save();raise
