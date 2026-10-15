"""Two-layer fusion diagnostic with exported-weight equality checked first."""
import argparse,json
from pathlib import Path
import gguf,numpy as np,torch
from trained_probe import TrainedDecoder,feed_for
from exported_int8 import apply_exported_int8
from probe import INPUTS
p=argparse.ArgumentParser()
for name in ('model','source','graph','prompt','output'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--variant',required=True);p.add_argument('--position',type=int,default=43)
a=p.parse_args();torch.set_num_threads(2)
c=json.loads((a.graph/'conversion.json').read_text());first=c['first_layer'];width=c['fusion_width'];capacity=c['capacity']
assert c['query_tokens']==1 and not c['key_cache_transposed']
r=gguf.GGUFReader(a.model);ids=json.loads(a.prompt.read_text())['token_ids'][:a.position+1]
assert len(ids)==a.position+1
t=next(t for t in r.tensors if t.name=='token_embd.weight');x=torch.from_numpy(gguf.dequantize(t.data,t.tensor_type)[ids][None].copy())
feeds={};outputs={};a.output.mkdir(exist_ok=False)
with torch.inference_mode():
    for i in range(first+width):
        l=TrainedDecoder(r,i,True,strict_mask=True).eval()
        apply_exported_int8(l,a.source/a.variant/f'l{i}')
        if i>=first:
            offset=i-first;fused=TrainedDecoder(r,i,True,strict_mask=True).eval()
            apply_exported_int8(fused,a.graph,prefix=f'b{offset}_')
            for key,v in l.state_dict().items():assert torch.equal(v,fused.state_dict()[key]),key
            _,k,v=l.project(x,*l.rope(torch.arange(len(ids))))
            pk=torch.zeros(1,l.kv,capacity,l.hd);pv=pk.clone()
            pk[:,:,:a.position]=k[:,:,:a.position];pv[:,:,:a.position]=v[:,:,:a.position]
            f=feed_for(l,x[:,-1:].numpy(),pk.numpy(),pv.numpy(),a.position,capacity)
            if offset==0:query=x[:,-1:].clone();feeds['b0_hidden']=f['hidden']
            feeds.update({f'b{offset}_'+n:f[n] for n in INPUTS if n!='hidden'})
            f['hidden']=query.numpy()
            query,newk,newv=l(*(torch.from_numpy(f[n]) for n in INPUTS))
            outputs[f'b{offset}_attach_new_k/out0']=newk.numpy()
            outputs[f'b{offset}_attach_new_v/out0']=newv.numpy()
        x=l.full(x)
outputs[f'b{width-1}_attach_hidden_out/out0']=query.numpy()
g=json.loads((a.graph/'decoder.json').read_text())['Layers']
ins=[g[k]['name'] for k in c['input_order']];outs=[g[k]['name'] for k in c['output_order']]
np.concatenate([feeds[n].ravel() for n in ins]).tofile(a.output/'probe-input.f32')
np.concatenate([outputs[n].ravel() for n in outs]).tofile(a.output/'probe-reference.f32')
for name in ('export_nbg_unify','conversion.json'):(a.output/name).symlink_to((a.graph/name).resolve())
(a.output/'probe-fixture.json').write_text(json.dumps({'scope':__doc__,'position':a.position,'exported_weights_identical':True,
    'outputs':[(n,outputs[n].size) for n in outs]},indent=2))
