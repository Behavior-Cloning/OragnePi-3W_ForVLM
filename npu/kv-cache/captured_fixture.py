"""Replay a captured graph call against independent exported-weight arithmetic."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import gguf
import numpy as np
import torch
from exported_int8 import apply_exported_int8
from graph_reference import evaluate
from trained_probe import TrainedDecoder

p=argparse.ArgumentParser()
p.add_argument('--model',type=Path,required=True)
p.add_argument('--graph-folder',type=Path,required=True)
p.add_argument('--capture',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();torch.set_num_threads(2)
a.output.mkdir(parents=True,exist_ok=False)
source=a.graph_folder.resolve()
for name in ('decoder.json','decoder.data','decoder.quantize','conversion.json','export_nbg_unify'):
    (a.output/name).symlink_to(source/name,target_is_directory=(source/name).is_dir())
conversion=json.loads((source/'conversion.json').read_text())
graph=json.loads((source/'decoder.json').read_text())
data=np.load(source/'decoder.data',allow_pickle=True).item()
raw=np.fromfile(a.capture,np.float32);feeds={};offset=0
for key in conversion['input_order']:
    node=graph['Layers'][key];shape=node['parameters']['shape'];count=int(np.prod(shape))
    feeds[node['name']]=torch.from_numpy(raw[offset:offset+count].reshape(shape).copy());offset+=count
assert offset==raw.size and np.isfinite(raw).all()
reader=gguf.GGUFReader(a.model)
width=conversion['fusion_width'];n=conversion['query_tokens']
def reference(exported):
    x=feeds['b0_hidden'];values={}
    for i in range(width):
        layer=TrainedDecoder(reader,conversion['first_layer']+i,True).eval()
        if exported:apply_exported_int8(layer,source,prefix=f'b{i}_')
        f={k:feeds[f'b{i}_'+k] for k in ('past_k','past_v','rope_cos','rope_sin')}
        if conversion.get('key_cache_transposed'):
            assert tuple(f['past_k'].shape[-2:]) == (layer.hd, conversion['capacity'])
            f['past_k']=f['past_k'].transpose(-2,-1)
        x,k,v=layer(x,f['past_k'],f['past_v'],feeds[f'b{i}_mask'][:,:,:n],f['rope_cos'],f['rope_sin'])
        values.update({f'b{i}_attach_new_k/out0':k,f'b{i}_attach_new_v/out0':v})
    values[f'b{width-1}_attach_hidden_out/out0']=x
    return values
with torch.inference_mode():
    original=reference(False);interpreted=evaluate(graph,data,feeds)
    graph_errors={}
    for key,value in interpreted.items():
        torch.testing.assert_close(value,original[key],atol=1e-4,rtol=5e-4)
        graph_errors[key]=float((value-original[key]).norm()/original[key].norm())
    expected=reference(True)
    outputs=[graph['Layers'][k]['name'] for k in conversion['output_order']]
    np.concatenate([expected[k].numpy().ravel() for k in outputs]).tofile(a.output/'probe-reference.f32')
    np.concatenate([original[k].numpy().ravel() for k in outputs]).tofile(a.output/'probe-original-reference.f32')
shutil.copyfile(a.capture,a.output/'probe-input.f32')
report={'scope':'actual captured NPU inputs, independent per-layer reference with exported constants',
    'capture_path':str(a.capture.resolve()),'capture_sha256':hashlib.sha256(a.capture.read_bytes()).hexdigest(),
    'capacity':conversion['capacity'],'query_tokens':n,'first_layer':conversion['first_layer'],
    'key_cache_transposed':conversion.get('key_cache_transposed',False),
    'fp32_graph_errors':graph_errors,'outputs':[(k,expected[k].numel()) for k in outputs]}
(a.output/'probe-fixture.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
