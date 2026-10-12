"""Verify batched graph shape/layout and build a real-history two-layer fixture."""
import argparse
import json
from pathlib import Path
import gguf
import numpy as np
import torch
from graph_reference import evaluate
from trained_probe import TrainedDecoder
from exported_int8 import apply_exported_int8

p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
p.add_argument('--source',type=Path,required=True);p.add_argument('--folder',type=Path,required=True)
p.add_argument('--history',type=int,default=84);a=p.parse_args();torch.set_num_threads(2)
conversion=json.loads((a.folder/'conversion.json').read_text());capacity=conversion['capacity'];n=conversion['query_tokens']
assert a.history+n<=capacity
m=json.loads((a.source/'trained-manifest.json').read_text());reader=gguf.GGUFReader(a.model)
embedding=next(t for t in reader.tensors if t.name=='token_embd.weight')
emb=gguf.dequantize(embedding.data,embedding.tensor_type)
tokens=json.loads((a.source/'prompt.json').read_text())['token_ids']
tokens=(tokens*((a.history+n)//len(tokens)+1))[:a.history+n]
hidden=torch.from_numpy(emb[tokens][None].copy())
graph=json.loads((a.folder/'decoder.json').read_text());data=np.load(a.folder/'decoder.data',allow_pickle=True).item()
feeds={};ref=[];prefix=conversion['first_layer'];width=conversion['fusion_width']
with torch.inference_mode():
    for i in range(prefix):hidden=TrainedDecoder(reader,i,True).eval().full(hidden)
    query=hidden[:,a.history:].clone()
    for offset in range(width):
        layer=TrainedDecoder(reader,prefix+offset,True).eval()
        _,keys,values=layer.project(hidden,*layer.rope(torch.arange(len(tokens))))
        pk=torch.zeros(1,layer.kv,capacity,layer.hd);pv=pk.clone()
        pk[:,:,:a.history]=keys[:,:,:a.history];pv[:,:,:a.history]=values[:,:,:a.history]
        mask=torch.full((1,1,n,capacity+n),-10000.)
        mask[:,:,:,:a.history]=0
        for j in range(n):mask[:,:,j,capacity:capacity+j+1]=0
        if n>1:mask=mask.repeat(1,1,layer.heads//layer.kv,1)
        cos,sin=layer.rope(torch.arange(a.history,a.history+n))
        feeds.update({f'b{offset}_'+k:v for k,v in dict(past_k=pk,past_v=pv,mask=mask,rope_cos=cos,rope_sin=sin).items()})
        if offset==0:feeds['b0_hidden']=query
        hidden=layer.full(hidden)
        ref.append((keys[:,:,a.history:],values[:,:,a.history:]))
    graph_feeds={k:(v.transpose(-1,-2).contiguous() if conversion.get('key_cache_transposed') and k.endswith('_past_k') else v)
                 for k,v in feeds.items()}
    outputs=evaluate(graph,data,graph_feeds)
    expected={f'b{width-1}_attach_hidden_out/out0':hidden[:,a.history:]}
    for i,(k,v) in enumerate(ref):expected.update({f'b{i}_attach_new_k/out0':k,f'b{i}_attach_new_v/out0':v})
    errors={}
    for name,actual in outputs.items():
        torch.testing.assert_close(actual,expected[name],atol=1e-4,rtol=5e-4)
        errors[name]=float((actual-expected[name]).norm()/expected[name].norm())
    assert max(errors.values())<1e-5
    # Re-evaluate with exported weights, supplying the same actual fixture
    # history per layer. This isolates board execution from weight encoding.
    x=query;board_expected={}
    for i in range(width):
        layer=TrainedDecoder(reader,prefix+i,True).eval();apply_exported_int8(layer,a.folder,prefix=f'b{i}_')
        f={k:feeds[f'b{i}_'+k] for k in ('past_k','past_v','rope_cos','rope_sin')}
        mask=feeds[f'b{i}_mask'][:,:,:n]
        x,k,v=layer(x,f['past_k'],f['past_v'],mask,f['rope_cos'],f['rope_sin'])
        board_expected.update({f'b{i}_attach_new_k/out0':k,f'b{i}_attach_new_v/out0':v})
    board_expected[f'b{width-1}_attach_hidden_out/out0']=x
    input_order=[graph['Layers'][k]['name'] for k in conversion['input_order']]
    output_order=[graph['Layers'][k]['name'] for k in conversion['output_order']]
    np.concatenate([graph_feeds[k].numpy().ravel() for k in input_order]).tofile(a.folder/'probe-input.f32')
    np.concatenate([board_expected[k].numpy().ravel() for k in output_order]).tofile(a.folder/'probe-reference.f32')
    report={'history':a.history,'capacity':capacity,'query_tokens':n,'fp32_graph_errors':errors,
            'key_cache_transposed':conversion.get('key_cache_transposed',False),
            'inputs':[(k,feeds[k].numel()) for k in input_order],
            'outputs':[(k,board_expected[k].numel()) for k in output_order]}
    (a.folder/'probe-fixture.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
