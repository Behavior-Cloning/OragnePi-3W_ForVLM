from pathlib import Path
import json,yaml,numpy as np,torch,sys,time,traceback
D=Path(__file__).resolve().parent;R=D.parent;sys.path.insert(0,str(R/'graphs/candidates-fp16'))
from graph_reference_conv import evaluate
torch.set_num_threads(2);S=R/'graphs/cases/c512-m32-l2';g=json.loads((S/'decoder.json').read_text());data=np.load(S/'decoder.data',allow_pickle=True).item();conv=json.loads((S/'conversion.json').read_text());q=yaml.safe_load((R/'graphs/candidates-down-i16-explicit/c512-m32-l2/decoder.quantize').read_text())['quantize_parameters']
st={'status':'running','started':time.time(),'scope':'FP32 source intermediates versus candidate ranges; not actual NPU internal values','rows':[]}
def save():(D/'range-diagnostic.json').write_text(json.dumps(st,indent=2)+'\n')
try:
 save()
 with torch.inference_mode():
  for c in json.loads((D/'status.json').read_text())['cases']:
   raw=np.fromfile(D/c['id']/'input.f32',dtype='<f4');offset=0;feeds={}
   for key in conv['input_order']:
    node=g['Layers'][key];shape=node['parameters']['shape'];n=int(np.prod(shape));feeds[node['name']]=torch.from_numpy(raw[offset:offset+n].copy().reshape(shape));offset+=n
   assert offset==raw.size
   _,cache=evaluate(g,data,feeds,return_intermediates=True);rows=[]
   for i in range(2):
    for key,qkey in [(f'b{i}_reshape_115',f'@down_i16_{2000+i}:out0'),(f'b{i}_fullconnect_116',f'@b{i}_fullconnect_116:out0')]:
     x=cache[key].numpy();fl=q[qkey]['fl'];codes=np.rint(x.astype(np.float64)*2**fl);sat=(codes>32767)|(codes<-32768)
     rows.append({'node':key,'fl':fl,'maxabs':float(np.max(abs(x))),'range_abs':32767/2**fl,'saturated_elements':int(sat.sum()),'elements':int(x.size),'affected_rows':int(np.any(sat.reshape(32,-1),axis=1).sum())})
   st['rows'].append({'id':c['id'],'ranges':rows});save();print(c['id'],flush=True)
 st.update(status='complete',ended=time.time());save()
except BaseException:st.update(status='failed',error=traceback.format_exc());save();raise
