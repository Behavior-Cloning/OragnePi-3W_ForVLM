import os,json,traceback
from pathlib import Path
os.environ['JEV_DYNAMIC_INT8']='1'
from board_adapters import Adapter
r=Path(__file__).resolve().parent;results=[]
for model in ['nano33','laya']:
 try:
  m=Adapter(model,r,2);z=m.decide('Please look to your left.',{'left':'Look left','right':'Look right','stop':'Stop'},'Which action matches the request?');results.append({'model':model,'config':m.optimization,'result':z,'status':'runnable_on_WSL_only'});m.close();del m
 except Exception as e:results.append({'model':model,'error':repr(e),'trace':traceback.format_exc()})
(r/'results/desktop-int8-compatibility.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));print(json.dumps(results,ensure_ascii=False,indent=2))
