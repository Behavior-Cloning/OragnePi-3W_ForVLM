#!/usr/bin/env python3
"""Read installed Acuity shape/schema. No graph writes, export, compilation or board."""
from pathlib import Path
from types import SimpleNamespace
import json,hashlib,inspect,subprocess
from acuitylib.layer.fullconnect import FullConnect
from acuitylib.core.shape import Shape
class ReadOnlyShape(FullConnect):
 def __init__(self,shape,axis,outputs):
  super().__init__('in_memory_shape_only',None);self.s=Shape(shape=shape);self.p=SimpleNamespace(weights=outputs,bias=True,regularize=False,axis=axis)
 @property
 def params(self):return self.p
 def get_in_shape(self,id=0):return self.s
 def get_out_shape(self,id=0):return self.compute_shape_nhwc()[id]
R=Path(__file__).resolve().parent;SRC=R.parent/'cases/c512-m32-l0';g=json.loads((SRC/'decoder.json').read_text());rows=[]
for key,node in g['Layers'].items():
 if node['op']!='fullconnect':continue
 pre=node['inputs'][0].split(':')[0][1:];ni=g['Layers'][pre]['parameters']['shape'][1];no=node['parameters']['weights'];row={'key':key,'I':ni,'O':no}
 for label,shape,axis in [('original',[32,ni],1),('axis3',[1,1,32,ni],3)]:
  p=ReadOnlyShape(shape,axis,no);result=p.compute_shape_nhwc()[0].dims;expect=shape[:-1]+[no];assert result==expect
  item={'input_shape':shape,'axis':axis,'output_shape':result}
  try:
   v=p.get_variable_shape('weight');item['weight_shape']=v.dims if hasattr(v,'dims') else v
  except Exception as e:item['weight_shape_not_checked']=type(e).__name__+': '+str(e)
  row[label]=item
 rows.append(row)
root=Path('/usr/local/lib/python3.8/dist-packages/acuitylib');bit_hits=[];scanned=0
for folder in [root/'layer',root/'app/exporter/ovxlib_case',root/'app/importer',root/'optimize']:
 for p in folder.rglob('*'):
  if p.is_file() and p.suffix in ['.so','.py','.json']:
   scanned+=1;b=p.read_bytes().lower()
   if b'bitcast' in b or b'bit_cast' in b:bit_hits.append(str(p))
files=[root/'layer/fullconnect.cpython-38-x86_64-linux-gnu.so',root/'app/exporter/ovxlib_case/vxnode_mapper.cpython-38-x86_64-linux-gnu.so',root/'app/exporter/timvx_case/timvx_schema/fullyconnected.json']
out={'scope':'SDK pure shape calls and file reads; no model export or graph compilation','fullconnect_axis_results':rows,'source_graph_sha256':hashlib.sha256((SRC/'decoder.json').read_bytes()).hexdigest(),'sdk_files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'bitcast_standard_frontend_search':{'files_scanned':scanned,'matches':bit_hits,'scope':['layer','app/exporter/ovxlib_case','app/importer','optimize'],'conclusion':'No standard native bitcast layer/mapper found in these installed modules; not proof against custom exporter plugins.'},'caveat':'Shape support does not prove a different compiled kernel, unchanged NPU rounding, or speedup. Original generated C uses legacy FCL without axis; exporter also has a vxFullconnectAxis path.'}
print(json.dumps(out,indent=2))
