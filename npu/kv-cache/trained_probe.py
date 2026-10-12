"""Export first two trained SmolVLM language layers with incremental KV.

GGUF Q8_0 is dequantized before ACUITY conversion; this is not native NPU Q8.
This checks layer arithmetic, not full-model language quality.
"""
import argparse
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.nn import functional as F
from probe import INPUTS, OUTPUTS


class TrainedDecoder(nn.Module):
    def __init__(self, reader, index):
        super().__init__()
        field = lambda name: reader.fields['llama.'+name].contents()
        self.dim = field('embedding_length')
        self.heads = field('attention.head_count')
        self.kv = field('attention.head_count_kv')
        self.hd = self.dim // self.heads
        self.base = field('rope.freq_base')
        self.eps = field('attention.layer_norm_rms_epsilon')
        tensors = {t.name: t for t in reader.tensors}
        for name, suffix in [('q','attn_q'), ('k','attn_k'), ('v','attn_v'),
                             ('o','attn_output'), ('gate','ffn_gate'),
                             ('up','ffn_up'), ('down','ffn_down'),
                             ('norm1','attn_norm'), ('norm2','ffn_norm')]:
            t = tensors[f'blk.{index}.{suffix}.weight']
            w = gguf.dequantize(t.data, t.tensor_type).copy()
            if name in ('q', 'k'):
                # GGUF llama rows use interleaved RoPE. Restore HF split-half rows.
                heads = self.heads if name == 'q' else self.kv
                w = w.reshape(heads, self.hd//2, 2, self.dim).swapaxes(1,2).reshape(w.shape).copy()
            self.register_buffer(name, torch.from_numpy(w))

    def rope(self, positions):
        freq = positions.float()[:,None] / self.base ** (torch.arange(self.hd//2).float()/(self.hd//2))
        angles = torch.cat((freq, freq), -1)[None,None]
        return angles.cos(), angles.sin()

    def norm(self, x, w):
        return x * torch.rsqrt((x*x).mean(-1, keepdim=True)+self.eps) * w

    def project(self, x, cos, sin):
        x = self.norm(x, self.norm1)
        q = F.linear(x, self.q).reshape(1,-1,self.heads,self.hd).transpose(1,2)
        k = F.linear(x, self.k).reshape(1,-1,self.kv,self.hd).transpose(1,2)
        v = F.linear(x, self.v).reshape(1,-1,self.kv,self.hd).transpose(1,2)
        rotate = lambda y: y*cos + torch.cat((-y[...,self.hd//2:],y[...,:self.hd//2]),-1)*sin
        return rotate(q), rotate(k), v

    def finish(self, hidden, attention):
        x = hidden + F.linear(attention.transpose(1,2).reshape(1,-1,self.dim), self.o)
        y = self.norm(x, self.norm2)
        return x + F.linear(F.silu(F.linear(y,self.gate))*F.linear(y,self.up), self.down)

    def forward(self, hidden, past_k, past_v, mask, rope_cos, rope_sin):
        q,k,v = self.project(hidden, rope_cos, rope_sin)
        keys = torch.cat((past_k,k),2).repeat_interleave(self.heads//self.kv,1)
        values = torch.cat((past_v,v),2).repeat_interleave(self.heads//self.kv,1)
        attention = torch.softmax(q @ keys.transpose(-1,-2)/self.hd**.5 + mask,-1) @ values
        return self.finish(hidden,attention),k,v

    def full(self, hidden):
        q,k,v = self.project(hidden,*self.rope(torch.arange(hidden.shape[1])))
        # Independent full-sequence causal attention is only a correctness oracle.
        attention = F.scaled_dot_product_attention(q,k.repeat_interleave(self.heads//self.kv,1),
                                                    v.repeat_interleave(self.heads//self.kv,1),is_causal=True)
        return self.finish(hidden, attention)


def feed_for(layer, hidden, k, v, pos, capacity):
    mask = np.full((1,1,1,capacity+1),-10000,np.float32)
    mask[...,:pos] = 0
    mask[...,-1] = 0
    cos,sin = layer.rope(torch.tensor([pos]))
    return dict(zip(INPUTS,[hidden,k,v,mask,cos.numpy(),sin.numpy()]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--capacity', type=int, default=128)
    args = p.parse_args()
    torch.set_num_threads(1)
    reader = gguf.GGUFReader(args.model)
    layers = [TrainedDecoder(reader,i).eval() for i in range(2)]
    layer = layers[0]
    args.output.mkdir(parents=True,exist_ok=True)
    emb = next(t for t in reader.tensors if t.name=='token_embd.weight')
    embeddings = gguf.dequantize(emb.data,emb.tensor_type)
    vocab = reader.fields['tokenizer.ggml.tokens'].contents()
    ids = {v:i for i,v in enumerate(vocab)}
    # Explicit valid BPE pieces; no unverified external tokenizer conversion.
    sequences = [
        ['Hello', ',', 'Ġhow', 'Ġare', 'Ġyou', '?', 'ĠI', 'Ġam', 'Ġhere', 'Ġto', 'Ġhelp', '.',
         'ĠWe', 'Ġcan', 'Ġtalk', 'Ġabout', 'Ġyour', 'Ġday', '.'],
        ['Today', 'ĠI', 'Ġfeel', 'Ġa', 'Ġlittle', 'Ġsad', '.', 'ĠCan', 'Ġyou', 'Ġstay',
         'Ġwith', 'Ġme', '?', 'ĠOf', 'Ġcourse', ',', 'ĠI', 'Ġwill', 'Ġlisten', '.']]
    report = {'model':args.model.name,'sha256':hashlib.file_digest(args.model.open('rb'),'sha256').hexdigest(),
              'layers':[0,1], 'capacity':args.capacity, 'dim':layer.dim,'heads':layer.heads,
              'kv_heads':layer.kv,'head_dim':layer.hd,'rope_base':layer.base,
              'scope':'first two trained layers only, no logits or language generation', 'cases':[]}
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    sessions=[]
    for i,l in enumerate(layers):
        cache=np.zeros((1,l.kv,args.capacity,l.hd),np.float32)
        feed=feed_for(l,np.ones((1,1,l.dim),np.float32),cache,cache,0,args.capacity)
        path=args.output/f'layer{i}.onnx'
        torch.onnx.export(l,tuple(torch.from_numpy(v) for v in feed.values()),str(path),
                          input_names=INPUTS,output_names=OUTPUTS,opset_version=13,dynamo=False)
        model=onnx.load(path);model.ir_version=8;onnx.checker.check_model(model,full_check=True);onnx.save(model,path)
        sessions.append(ort.InferenceSession(str(path),opts,providers=['CPUExecutionProvider']))
    with torch.inference_mode():
        for case,pieces in enumerate(sequences):
            token_ids=([ids[v] for v in pieces]*((args.capacity//len(pieces))+1))[:args.capacity]
            hidden=embeddings[token_ids][None].copy()
            oracle=torch.from_numpy(hidden)
            for l in layers:oracle=l.full(oracle)
            if case==1:
                hidden.tofile(args.output/'hidden.f32');oracle.numpy().tofile(args.output/'reference.f32')
            caches=np.zeros((2,2,1,layer.kv,args.capacity,layer.hd),np.float32)
            datasets=[[],[]];max_error=0.
            for pos in range(args.capacity):
                x=hidden[:,pos:pos+1]
                for i,session in enumerate(sessions):
                    feed=feed_for(layers[i],x,caches[i,0],caches[i,1],pos,args.capacity)
                    if case==0 and pos in {0,1,2,4,8,args.capacity//2,args.capacity-1}:
                        names=[]
                        for name,value in feed.items():
                            filename=f'l{i}_p{pos}_{name}.npy';np.save(args.output/filename,value);names.append(filename)
                        datasets[i].append(' '.join(names))
                    x,k,v=session.run(OUTPUTS,feed)
                    caches[i,0,:,:,pos:pos+1]=k;caches[i,1,:,:,pos:pos+1]=v
                expected=oracle[:,pos:pos+1].numpy()
                max_error=max(max_error,float(np.max(np.abs(x-expected))))
                np.testing.assert_allclose(x,expected,atol=3e-5,rtol=3e-4)
            if case==0:
                for i,entries in enumerate(datasets):(args.output/f'layer{i}-dataset.txt').write_text('\n'.join(entries)+'\n')
            report['cases'].append({'case':case,'bpe_pieces':pieces,'tokens':len(token_ids),
                                    'max_abs_error':max_error,'host_ort_check':'PASS'})
    (args.output/'trained-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
