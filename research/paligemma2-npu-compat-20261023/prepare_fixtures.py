from pathlib import Path
import numpy as np,hashlib,json,tarfile
r=Path(__file__).resolve().parent;b=Path('/opt/opi-lab/paligemma2-npu-compat-20261023');d=b/'board-fixtures';d.mkdir(exist_ok=True)
names=['hidden','past_k','past_v','mask','rope_cos','rope_sin'];manifest=[]
for i in range(18):
 inputs=np.concatenate([np.load(b/f's{i}_{n}.npy').reshape(-1) for n in names]).astype('<f4');inputs.tofile(d/f'p{i}-inputs.f32');np.load(b/f's{i}_reference.npy').astype('<f4').tofile(d/f'p{i}-reference.f32')
for f in d.iterdir():manifest.append(dict(name=f.name,bytes=f.stat().st_size,sha256=hashlib.file_digest(f.open('rb'),'sha256').hexdigest()))
(d/'manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(b/'board-fixtures.tar','w') as tf:
 for f in d.iterdir():tf.add(f,arcname=f.name)
(r/'fixtures-manifest.json').write_text(json.dumps(manifest,indent=2))
