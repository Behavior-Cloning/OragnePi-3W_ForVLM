import torch,time,json,psutil,os
from pathlib import Path
torch.set_num_threads(2);torch.set_num_interop_threads(1)
t=time.perf_counter();import laya
m=laya.load('/opt/opi-vlm/jev-lab-20261020/models/laya',device='cpu');print('load_seconds',time.perf_counter()-t,flush=True)
q={'action':{'type':'choice','instructions':'选择符合用户要求的安全动作。','criteria':{'stop':'立即停止','left':'看向左侧','right':'看向右侧','help':'请求帮助'}}}
for i in range(3):
 t=time.perf_counter();r=m.predict({'user':'请看向左侧','scene':'客厅'},q,head_max_len=512,max_len=1024);print(json.dumps({'elapsed':time.perf_counter()-t,'result':r,'rss':psutil.Process().memory_info().rss},ensure_ascii=False),flush=True)
