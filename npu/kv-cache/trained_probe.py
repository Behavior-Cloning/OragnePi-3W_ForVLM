"""Export first two trained llama/Qwen2 language layers with incremental KV.

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


def rms_norm(x, weight, eps, stable=False, version=3):
    epsilon=eps
    if stable:
        divisor=x.abs().amax(-1,keepdim=True).clamp_min(eps**0.5 if version==3 else 1.0)
        x=x/divisor
        # Avoid materializing eps=1e-5 as a subnormal FP16 divide numerator.
        epsilon=(eps**0.5/divisor).square() if version==3 else eps/divisor/divisor
    return x*torch.rsqrt((x*x).mean(-1,keepdim=True)+epsilon)*weight


class LMHead(nn.Module):
    def __init__(self, reader, eps, stable_norm=False, norm_version=3):
        super().__init__()
        tensors={t.name:t for t in reader.tensors}
        for name,key in [('scale','output_norm.weight'),('weight','output.weight')]:
            t=tensors.get(key,tensors['token_embd.weight'])
            self.register_buffer(name,torch.from_numpy(gguf.dequantize(t.data,t.tensor_type).copy()))
        self.eps=eps
        self.stable_norm=stable_norm
        self.norm_version=norm_version

    def forward(self,hidden):
        normalized=rms_norm(hidden,self.scale,self.eps,self.stable_norm,self.norm_version)
        return F.linear(normalized,self.weight)


class TrainedDecoder(nn.Module):
    def __init__(self, reader, index, stable_norm=False, norm_version=3):
        super().__init__()
        self.stable_norm=stable_norm
        self.norm_version=norm_version
        self.arch = reader.fields['general.architecture'].contents()
        if self.arch not in ('llama', 'qwen2', 'qwen3', 'qwen2vl'):
            raise ValueError(f'Unsupported architecture: {self.arch}')
        field = lambda name: reader.fields[self.arch+'.'+name].contents()
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
            if self.arch == 'llama' and name in ('q', 'k'):
                # GGUF llama rows use interleaved RoPE. Restore HF split-half rows.
                heads = self.heads if name == 'q' else self.kv
                w = w.reshape(heads, self.hd//2, 2, self.dim).swapaxes(1,2).reshape(w.shape).copy()
            self.register_buffer(name, torch.from_numpy(w))
        for name in ('q','k','v'):
            t=tensors.get(f'blk.{index}.attn_{name}.bias')
            bias=torch.from_numpy(gguf.dequantize(t.data,t.tensor_type).copy()) if t is not None else None
            if bias is not None and self.arch == 'llama':
                raise ValueError('llama QKV bias layout not implemented')
            self.register_buffer(name+'_bias',bias)
        for name in ('q','k'):
            t=tensors.get(f'blk.{index}.attn_{name}_norm.weight')
            weight=torch.from_numpy(gguf.dequantize(t.data,t.tensor_type).copy()) if t is not None else None
            self.register_buffer(name+'_norm',weight)

    def rope(self, positions):
        freq = positions.float()[:,None] / self.base ** (torch.arange(self.hd//2).float()/(self.hd//2))
        angles = torch.cat((freq, freq), -1)[None,None]
        return angles.cos(), angles.sin()

    def norm(self, x, w):
        return rms_norm(x,w,self.eps,self.stable_norm,self.norm_version)

    def project(self, x, cos, sin):
        x = self.norm(x, self.norm1)
        q = F.linear(x, self.q, self.q_bias).reshape(1,-1,self.heads,self.hd).transpose(1,2)
        k = F.linear(x, self.k, self.k_bias).reshape(1,-1,self.kv,self.hd).transpose(1,2)
        v = F.linear(x, self.v, self.v_bias).reshape(1,-1,self.kv,self.hd).transpose(1,2)
        if self.q_norm is not None:q=self.norm(q,self.q_norm)
        if self.k_norm is not None:k=self.norm(k,self.k_norm)
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
    p.add_argument('--layers', type=int, default=2, help='0 exports all language decoder layers')
    p.add_argument('--tokens', type=int, help='defaults to capacity')
    p.add_argument('--stable-norm',action='store_true')
    p.add_argument('--norm-version',type=int,choices=(1,3),default=3)
    args = p.parse_args()
    torch.set_num_threads(1)
    reader = gguf.GGUFReader(args.model)
    arch=reader.fields['general.architecture'].contents()
    total_layers=reader.fields[arch+'.block_count'].contents()
    count=args.layers or total_layers
    tokens=args.tokens or args.capacity
    if not 1<=count<=total_layers or not 1<=tokens<=args.capacity:raise ValueError('invalid layers/tokens')
    layers = [TrainedDecoder(reader,i,stable_norm=args.stable_norm,norm_version=args.norm_version).eval() for i in range(count)]
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
    if count==total_layers:
        sequences[0]=['<|im_start|>','User',':','ĠHello','!','<end_of_utterance>','Ċ','Ass','istant',':']+sequences[0]
    report = {'model':args.model.name,'sha256':hashlib.file_digest(args.model.open('rb'),'sha256').hexdigest(),
              'architecture':layer.arch,'stable_norm':args.stable_norm,'norm_version':args.norm_version if args.stable_norm else None,'language_layers_total':reader.fields[layer.arch+'.block_count'].contents(),
              'gguf_tensor_elements':sum(int(np.prod(t.shape)) for t in reader.tensors),
              'tested_layer_parameters':sum(b.numel() for l in layers for b in l.buffers()),
              'layers':list(range(count)), 'capacity':args.capacity, 'tokens':tokens, 'dim':layer.dim,'heads':layer.heads,
              'kv_heads':layer.kv,'head_dim':layer.hd,'rope_base':layer.base,
              'scope':f'{count} trained decoder layers, no final norm/logits or language generation',
              'host_tolerance':{'atol':5e-4 if layer.arch=='qwen2vl' else (1e-4 if count==2 else 1e-3),'rtol':3e-4,'relative_l2':1e-4},'cases':[]}
    head=LMHead(reader,layer.eps,stable_norm=args.stable_norm,norm_version=args.norm_version).eval() if count==total_layers else None
    if head is not None:
        report['scope']='all decoder layers plus final RMSNorm/LM head; board validation pending'
        report['vocab_size']=len(vocab)
        report['eos_token_ids']=[v.contents() for k,v in reader.fields.items()
                                 if k in ('tokenizer.ggml.eos_token_id','tokenizer.ggml.eot_token_id')]
        embeddings.tofile(args.output/'embeddings.f32')
        (args.output/'vocabulary.json').write_text(json.dumps(vocab,ensure_ascii=False))
        path=args.output/f'layer{count}.onnx'
        torch.onnx.export(head,torch.ones(1,1,layer.dim),str(path),input_names=['hidden'],
                          output_names=['logits'],opset_version=13,dynamo=False)
        model=onnx.load(path);model.ir_version=8;onnx.checker.check_model(model,full_check=True);onnx.save(model,path)
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
            token_ids=([ids[v] for v in pieces]*((tokens//len(pieces))+1))[:tokens]
            hidden=embeddings[token_ids][None].copy()
            oracle=torch.from_numpy(hidden)
            for l in layers:oracle=l.full(oracle)
            if head is not None and case==0:
                entries=[]
                for pos in sorted({0,1,2,4,8,tokens//2,tokens-1}):
                    if pos>=tokens:continue
                    filename=f'head_p{pos}_hidden.npy';np.save(args.output/filename,oracle[:,pos:pos+1].numpy());entries.append(filename)
                (args.output/f'layer{count}-dataset.txt').write_text('\n'.join(entries)+'\n')
            if case==1:
                hidden.tofile(args.output/'hidden.f32');oracle.numpy().tofile(args.output/'reference.f32')
            caches=np.zeros((count,2,1,layer.kv,args.capacity,layer.hd),np.float32)
            datasets=[[] for _ in layers];max_error=0.
            for pos in range(tokens):
                x=hidden[:,pos:pos+1]
                for i,session in enumerate(sessions):
                    feed=feed_for(layers[i],x,caches[i,0],caches[i,1],pos,args.capacity)
                    if case==0 and pos in {0,1,2,4,8,tokens//2,tokens-1}:
                        names=[]
                        for name,value in feed.items():
                            filename=f'l{i}_p{pos}_{name}.npy';np.save(args.output/filename,value);names.append(filename)
                        datasets[i].append(' '.join(names))
                    x,k,v=session.run(OUTPUTS,feed)
                    caches[i,0,:,:,pos:pos+1]=k;caches[i,1,:,:,pos:pos+1]=v
                expected=oracle[:,pos:pos+1].numpy()
                max_error=max(max_error,float(np.max(np.abs(x-expected))))
                # FP32 SDPA vs explicit ONNX reductions can differ near zero by ~4e-5
                # on Qwen. Also require a tight scale-normalized error bound.
                np.testing.assert_allclose(x,expected,atol=report['host_tolerance']['atol'],rtol=3e-4)
                relative_error=float(np.linalg.norm(x-expected)/max(np.linalg.norm(expected),1e-12))
                assert relative_error<report['host_tolerance']['relative_l2'], (case,pos,relative_error,float(np.max(np.abs(x-expected))))
            if case==0:
                for i,entries in enumerate(datasets):(args.output/f'layer{i}-dataset.txt').write_text('\n'.join(entries)+'\n')
            report['cases'].append({'case':case,'bpe_pieces':pieces,'tokens':len(token_ids),
                                    'max_abs_error':max_error,'host_ort_check':'PASS'})
    (args.output/'trained-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
