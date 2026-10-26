from pathlib import Path
import numpy as np,json,hashlib
from PIL import Image
p=Path(__file__).resolve().parent.parent/'fixtures';meta=[]
for name in ['image1','image2']:
 im=Image.open(p/(name+'.jpg')).convert('RGB').resize((448,448),Image.Resampling.BICUBIC)
 a=(np.asarray(im,dtype=np.float32)/255.0-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
 a.astype('<f4').tofile(p/(name+'.f32'));meta.append({'name':name,'shape':list(a.shape),'layout':'HWC RGB ImageNet normalization','sha256':hashlib.sha256((p/(name+'.f32')).read_bytes()).hexdigest()})
(p/'prepared.json').write_text(json.dumps(meta,indent=2))
