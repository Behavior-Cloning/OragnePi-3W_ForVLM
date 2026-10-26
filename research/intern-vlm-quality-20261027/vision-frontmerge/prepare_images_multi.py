from pathlib import Path
import numpy as np,json,hashlib
from PIL import Image
p=Path(__file__).resolve().parent.parent/'fixtures';meta=[]
for name in ['image1','image2']:
 for side in [112,224,448]:
  im=Image.open(p/(name+'.jpg')).convert('RGB').resize((side,side),Image.Resampling.BICUBIC)
  a=(np.asarray(im,dtype=np.float32)/255.0-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
  out=p/(name+'-'+str(side)+'.f32');a.astype('<f4').tofile(out)
  meta.append({'name':name,'side':side,'shape':list(a.shape),'layout':'HWC RGB ImageNet normalization','sha256':hashlib.sha256(out.read_bytes()).hexdigest()})
(p/'prepared-multi.json').write_text(json.dumps(meta,indent=2))
