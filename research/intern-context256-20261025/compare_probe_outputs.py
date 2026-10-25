#!/usr/bin/env python3
"""Compare all current-query outputs; no padded history tensor is output."""
from pathlib import Path
import argparse,hashlib,json,numpy as np
p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--new',type=Path,required=True);p.add_argument('--tokens',type=int,choices=[1,16],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
x=np.fromfile(a.old,dtype='<f4');y=np.fromfile(a.new,dtype='<f4');assert x.size==y.size==(896+4*128)*a.tokens
assert np.isfinite(x).all() and np.isfinite(y).all()
rows=[];off=0
for name,n in [('hidden',896),('layer0_new_k',128),('layer1_new_k',128),('layer0_new_v',128),('layer1_new_v',128)]:
 n*=a.tokens;u=x[off:off+n];v=y[off:off+n];e=v.astype('float64')-u;off+=n
 rows.append({'name':name,'elements':n,'byte_equal':u.tobytes()==v.tobytes(),'relative_l2':float(np.linalg.norm(e)/max(np.linalg.norm(u.astype('float64')),1e-30)),'max_abs':float(abs(e).max()),'old_sha256':hashlib.sha256(u.tobytes()).hexdigest(),'new_sha256':hashlib.sha256(v.tobytes()).hexdigest()})
report={'scope':'same-query single fused 2-layer block; not full-model task accuracy','old':str(a.old),'new':str(a.new),'query_tokens':a.tokens,'all_byte_equal':x.tobytes()==y.tobytes(),'outputs':rows}
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
