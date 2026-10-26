from pathlib import Path
import json,hashlib,re,time,traceback,os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np,torch,yaml
R=Path(__file__).resolve().parent;T=R.parent
state={'status':'running','pid':os.getpid(),'start':time.time()}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save():(R/'status.json').write_text(json.dumps(state,indent=2))
save()
try:
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 C=T/'results/down-capture-c1280-replay-r1';cfg=json.loads((C/'config.json').read_text());ev={e['case']:e for e in map(json.loads,(C/'events.jsonl').read_text().splitlines()) if e['event']=='result'}
 S=T/'graphs/down-capture-c1280-m16-l2';Q=T/'graphs/candidates-down-i16-c1280-range16/c1280-m16-l2';qp=yaml.safe_load((Q/'decoder.quantize').read_text())['quantize_parameters']
 def weight(folder,b):
  code=(folder/'export/vnn_decoder.c').read_text();i=code.index('/* @b%d_fullconnect_116:weight */'%b);j=code.index('/*',i+3);body=code[i:j]
  m=re.search(r'NEW_CONST_TENSOR\(const_tensor\[\d+\], attr, VSI_NN_TYPE_(FLOAT16|INT16), (\d+), (\d+)\)',body);assert m
  typ,off,n=m.groups();raw=(folder/'export/decoder.export.data').read_bytes()[int(off):int(off)+int(n)];assert len(raw)==4864*896*2
  w=np.frombuffer(raw,'<f2' if typ=='FLOAT16' else '<i2').reshape(896,4864).T.astype('float32').copy()
  fl=int(re.search(r'attr.dtype.fl = (\d+);',body).group(1)) if typ=='INT16' else 0
  return torch.from_numpy(w)/float(2**fl),{'type':typ,'fl':fl,'sha256':hashlib.sha256(raw).hexdigest()}
 values=[[],[]];observed=[[],[]];rows=[];capture_hashes={}
 for f in cfg['fixtures']:
  key=f['key']+'-capture';path=C/(key+'.f32');h=sha(path);assert h==ev[key]['output_sha256'];capture_hashes[path.name]=h
  a=np.fromfile(path,'<f4');assert a.size==16*(896+4*128)+32*4864+32*896 and np.isfinite(a).all();o=16*(896+4*128)
  for b in range(2):values[b].append(a[o+b*16*4864:o+(b+1)*16*4864].reshape(16,4864)[:f['valid_rows']].copy())
  o+=32*4864
  for b in range(2):observed[b].append(a[o+b*16*896:o+(b+1)*16*896].reshape(16,896)[:f['valid_rows']].copy())
  static=64 if f['case'].startswith('s64-') else 128
  for i in range(f['valid_rows']):rows.append({'case':f['case'],'pos':f['position']+i,'dynamic':f['position']+i>=static})
 cases=sorted(set(x['case'] for x in rows));assert len(cases)==2
 calcase=next(c for c in cases if c.startswith('s64-'));cal=torch.tensor([r['case']==calcase for r in rows]);held=~cal
 out={'scope':'Offline original-NPU activation captures: calibration only s64 prompt, heldout s128 prompt. Both are reused historical fixtures, not independent natural-language quality set. FP32 math plus explicit FP16 branch/output rounding; no NPU runtime or performance evidence.','cases':cases,'calibration_case':calcase,'rows':len(rows),'capture_sha256':capture_hashes,'layers':[]}
 def metrics(v,ref,npu,mask):
  a=v[mask];b=ref[mask];c=npu[mask]
  return {'rows':int(mask.sum()),'rel_l2_fp32':float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b)),'rel_l2_npu':float(torch.linalg.vector_norm(a-c)/torch.linalg.vector_norm(c)),'max_abs_npu':float((a-c).abs().max()),'finite':bool(torch.isfinite(a).all())}
 with torch.inference_mode():
  for b in range(2):
   x=torch.from_numpy(np.concatenate(values[b]));npu=torch.from_numpy(np.concatenate(observed[b]));w,wm=weight(S,b);wq,wqm=weight(Q,b);fl=qp['@down_i16_%d:out0'%(2000+b)]['fl'];ref=x@w
   peak=x[cal].abs().amax(0);order=torch.argsort(peak,descending=True);limits=[0,8,16,32,64,128,256,512,1024]
   masks={'cal_all':cal,'heldout_all':held,'heldout_first':held&torch.tensor([r['pos']==0 for r in rows]),'heldout_dynamic':held&torch.tensor([r['dynamic'] for r in rows])}
   xq=(x*2**fl).round().clamp(-32768,32767)/2**fl;base=xq@wq;rr=[]
   for k in limits:
    ids=order[:k];v=base.clone()
    if k:v=v-xq[:,ids]@wq[ids]+x[:,ids]@w[ids]
    # Preserve FP16 rounding at each branch sum and final result.
    fast=base-(xq[:,ids]@wq[ids] if k else 0);slow=x[:,ids]@w[ids] if k else torch.zeros_like(base)
    rounded=(fast.half().float()+slow.half().float()).half().float()
    remaining=torch.ones(4864,dtype=torch.bool);remaining[ids]=False
    rr.append({'fp16_channels':k,'ratio':k/4864,'channel_ids':ids.tolist(),'heldout_remaining_max':float(x[held][:,remaining].abs().max()),'heldout_clipped_entries':int(((x[held][:,remaining]*2**fl).abs()>32767).sum()),'ideal_branch_metrics':{n:metrics(v,ref,npu,m) for n,m in masks.items()},'fp16_branch_metrics':{n:metrics(rounded,ref,npu,m) for n,m in masks.items()}})
   thresholds={str(t):{'cal_channels':int((peak>t).sum()),'heldout_channels':int((x[held].abs().amax(0)>t).sum())} for t in [1,4,8,16,32,64,128,512,1024]}
   row={'layer':b+2,'original_weight':wm,'integer_weight':wqm,'activation_fl':fl,'threshold_channels':thresholds,'splits':rr}
   out['layers'].append(row);print('LAYER',b+2,'thresholds',thresholds,flush=True)
   print('SPLITS',[(a['fp16_channels'],a['heldout_remaining_max'],a['fp16_branch_metrics']['heldout_first']['rel_l2_npu'],a['fp16_branch_metrics']['heldout_dynamic']['rel_l2_npu']) for a in rr],flush=True)
 out['sources_sha256']={str(f.relative_to(T)):sha(f) for f in [Path(__file__),C/'config.json',C/'events.jsonl',S/'export/vnn_decoder.c',S/'export/decoder.export.data',Q/'export/vnn_decoder.c',Q/'export/decoder.export.data',Q/'decoder.quantize']}
 (R/'ANALYSIS.json').write_text(json.dumps(out,indent=2));state.update(status='complete',ended=time.time(),analysis_sha256=sha(R/'ANALYSIS.json'));save()
except BaseException:state.update(status='failed',error=traceback.format_exc(),ended=time.time());save();raise
