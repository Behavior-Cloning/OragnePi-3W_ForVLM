import sys,torch,numpy as np,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent));from model_layers import *
torch.set_grad_enabled(False);l=load_layer(0);a=l.linear_attn
r=Path(__file__).resolve().parents[1];raw=np.fromfile(r/'results/trace-linear0-output.f32',np.float32);ref=np.fromfile(ROOT/'trace-linear0/reference.f32',np.float32)
normin=torch.from_numpy(raw[:1024].copy()).reshape(1,1,1024);core=torch.from_numpy(raw[1024:3072].copy()).reshape(16,128);z=a.in_proj_z(normin).reshape(16,128)
gn=a.norm(core,z).numpy();npu=raw[3072:5120].reshape(16,128);orig=ref[3072:5120].reshape(16,128)
report={'gated_error_using_npu_core_and_cpu_z':float(np.linalg.norm(npu-gn)/np.linalg.norm(gn)),'heads':[]}
for h in range(16):report['heads'].append({'head':h,'gn_npu_rms':float(np.sqrt((npu[h]**2).mean())),'gn_cpu_rms':float(np.sqrt((orig[h]**2).mean())),'norm_from_same_core_rms':float(np.sqrt((gn[h]**2).mean())),'cosine':float(np.dot(npu[h],gn[h])/(np.linalg.norm(npu[h])*np.linalg.norm(gn[h])+1e-30))})
(r/'results/gated-norm-diagnosis.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
