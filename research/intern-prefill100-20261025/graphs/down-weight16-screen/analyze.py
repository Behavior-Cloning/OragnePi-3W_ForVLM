from pathlib import Path
import numpy as np,json,hashlib,math
T=Path(__file__).resolve().parents[2]
p=T/'graphs/cases/c512-m32-l2/decoder.data'
d=np.load(p,allow_pickle=True).item();rows=[]
for name in ['b0_fullconnect_116','b1_fullconnect_116']:
 w=d[name]['weight'].astype(np.float16).astype(np.float64)
 assert np.isfinite(w).all()
 for bits in [8,16]:
  limit=2**(bits-1)-1;fl=math.floor(math.log2(limit/float(np.max(abs(w)))))
  q=np.rint(w*2**fl);sat=int(np.sum((q>limit)|(q<-limit-1)));q=np.clip(q,-limit-1,limit)
  y=q/(2**fl);diff=y-w
  rows.append(dict(node=name,bits=bits,shape=list(w.shape),fl=fl,saturation_count=sat,relative_l2=float(np.linalg.norm(diff)/np.linalg.norm(w)),max_abs_error=float(np.max(abs(diff))),rounded_zero_fraction=float(np.mean(q==0)),fp16_source_sha256=hashlib.sha256(w.astype(np.float16).tobytes()).hexdigest()))
out=dict(scope='offline weight-only numeric screen; FP32 decoder weight rounded to FP16, not yet proved exported payload parity; no activation calibration, SDK compilation, board inference or full-model quality',source=str(p.resolve()),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rows=rows)
Path(__file__).with_name('analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
