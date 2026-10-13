"""Cross-check the full-depth custom oracle against standard Transformers.

Weights are independently mapped from the existing Q8 GGUF, not trained BF16.
Run on the workstation: these numbers are correctness, never board CPU speed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np
import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

from trained_probe import TrainedDecoder, LMHead


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def errors(actual, expected):
    a=actual.double().reshape(-1);b=expected.double().reshape(-1)
    assert torch.isfinite(a).all() and torch.isfinite(b).all() and a.shape==b.shape
    return {'relative_l2':float((a-b).norm()/b.norm().clamp_min(1e-20)),
            'max_abs':float((a-b).abs().max()),
            'cosine':float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-20))}


def load_standard(reader):
    arch=reader.fields['general.architecture'].contents();assert arch in ('llama','qwen2')
    f=lambda key:reader.fields[arch+'.'+key].contents()
    tensors={t.name:t for t in reader.tensors}
    def weight(name):
        t=tensors[name]
        return torch.from_numpy(gguf.dequantize(t.data,t.tensor_type).copy())
    dim=f('embedding_length');heads=f('attention.head_count');kv=f('attention.head_count_kv');hd=dim//heads
    layers=f('block_count');vocab=int(tensors['token_embd.weight'].shape[1])
    cls,model_cls=(LlamaConfig,LlamaForCausalLM) if arch=='llama' else (Qwen2Config,Qwen2ForCausalLM)
    config=cls(vocab_size=vocab,hidden_size=dim,intermediate_size=f('feed_forward_length'),
               num_hidden_layers=layers,num_attention_heads=heads,num_key_value_heads=kv,
               rms_norm_eps=f('attention.layer_norm_rms_epsilon'),rope_theta=f('rope.freq_base'),
               max_position_embeddings=f('context_length'),hidden_act='silu',
               attention_dropout=0.0,tie_word_embeddings='output.weight' not in tensors,
               use_sliding_window=False)
    config._attn_implementation='sdpa'
    with torch.device('meta'):
        model=model_cls(config)
    state={'model.embed_tokens.weight':weight('token_embd.weight'),
           'model.norm.weight':weight('output_norm.weight'),
           'lm_head.weight':weight('output.weight' if 'output.weight' in tensors else 'token_embd.weight')}
    used={'token_embd.weight','output_norm.weight'}|({'output.weight'} if 'output.weight' in tensors else set())
    mapping={'attn_q':'self_attn.q_proj','attn_k':'self_attn.k_proj','attn_v':'self_attn.v_proj',
             'attn_output':'self_attn.o_proj','ffn_gate':'mlp.gate_proj','ffn_up':'mlp.up_proj',
             'ffn_down':'mlp.down_proj','attn_norm':'input_layernorm','ffn_norm':'post_attention_layernorm'}
    for i in range(layers):
        for source,target in mapping.items():
            name=f'blk.{i}.{source}.weight';value=weight(name);used.add(name)
            if arch=='llama' and source in ('attn_q','attn_k'):
                # GGUF stores interleaved RoPE rows; HF uses split-half rows.
                count=heads if source=='attn_q' else kv
                value=value.view(count,hd//2,2,dim).transpose(1,2).reshape_as(value).contiguous()
            state[f'model.layers.{i}.{target}.weight']=value
            bias=f'blk.{i}.{source}.bias'
            if bias in tensors:
                assert arch=='qwen2' and source in ('attn_q','attn_k','attn_v')
                state[f'model.layers.{i}.{target}.bias']=weight(bias);used.add(bias)
    assert used==set(tensors), 'unmapped GGUF language tensors: '+str(set(tensors)-used)
    model.load_state_dict(state,strict=True,assign=True)
    # Rotary inv_freq is a nonpersistent buffer, absent from state_dict.
    # Recreate it through the standard implementation on the CPU.
    model.model.rotary_emb=type(model.model.rotary_emb)(config=config,device=torch.device('cpu'))
    model.eval();assert all(p.device.type=='cpu' and p.dtype==torch.float32 for p in model.parameters())
    assert all(b.device.type=='cpu' for b in model.buffers())
    return model,config


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True,help='full NPU run directory with prompt and frozen feedback')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    torch.set_num_threads(2)
    runtime=json.loads((a.reference/'tokens.json').read_text())
    prompt=json.loads((a.reference/'prompt.json').read_text())
    assert runtime['fixed_feedback'] and runtime['generated_tokens']==128
    ids=prompt['token_ids']+runtime['feedback_tokens'][:-1]
    history=runtime['cached_prefix_tokens'];new=len(prompt['token_ids'])-history
    reader=gguf.GGUFReader(a.model);model,config=load_standard(reader)
    assert len(model.model.layers)==runtime['decoder_layers']
    inputs=torch.tensor([ids],dtype=torch.long);captured=[]
    handles=[layer.register_forward_hook(lambda module, args, out:captured.append((out[0] if isinstance(out,tuple) else out).detach())) for layer in model.model.layers]
    with torch.inference_mode():
        standard=model.model(inputs,use_cache=False).last_hidden_state
        for handle in handles:handle.remove()
        standard_logits=model.lm_head(standard[:,len(prompt['token_ids'])-1:])
        assert len(captured)==config.num_hidden_layers
        custom=model.model.embed_tokens(inputs);layer_errors=[]
        for i in range(config.num_hidden_layers):
            layer=TrainedDecoder(reader,i).eval();custom=layer.full(custom)
            layer_errors.append({'layer':i,**errors(custom,captured[i])})
        custom_logits=LMHead(reader,config.rms_norm_eps).eval()(custom[:,len(prompt['token_ids'])-1:])
        oracle_error=errors(custom_logits,standard_logits)
        # Exercise persistent KV across history, a new message, and every
        # scalar feedback token, using the standard library cache machinery.
        preparation=model.model(inputs[:,:history],use_cache=True)
        cache=preparation.past_key_values
        first=model.model(inputs[:,history:history+new],past_key_values=cache,use_cache=True)
        cache=first.past_key_values;pieces=[model.lm_head(first.last_hidden_state[:,-1:])]
        for pos in range(history+new,len(ids)):
            next_step=model.model(inputs[:,pos:pos+1],past_key_values=cache,use_cache=True)
            cache=next_step.past_key_values;pieces.append(model.lm_head(next_step.last_hidden_state))
        cached_logits=torch.cat(pieces,dim=1)
        cache_error=errors(cached_logits,standard_logits)
        npu=torch.from_numpy(np.fromfile(a.reference/'logits.f32',np.float32).reshape(1,128,config.vocab_size))
        npu_error=errors(npu,standard_logits)
        std_logp=standard_logits.log_softmax(-1);npu_logp=npu.log_softmax(-1)
        kl=float((std_logp.exp()*(std_logp-npu_logp)).sum(-1).mean())
    passed=oracle_error['relative_l2']<=1e-4 and cache_error['relative_l2']<=1e-4 and all(e['relative_l2']<=1e-4 for e in layer_errors)
    report={'scope':'independent standard-library FP32 using dequantized GGUF Q8; workstation only; not original BF16 quality or CPU speed',
            'transformers_version':transformers.__version__,'torch_version':torch.__version__,
            'model_sha256':sha(a.model),'script_sha256':sha(Path(__file__)),
            'custom_oracle_source_sha256':sha(Path(__file__).with_name('trained_probe.py')),
            'reference':str(a.reference),'npu_logits_sha256':sha(a.reference/'logits.f32'),
            'config':config.to_dict(),'layers':config.num_hidden_layers,'history':history,'new_tokens':new,
            'generated_predictions':128,'layer_errors':layer_errors,'custom_vs_standard':oracle_error,
            'standard_scalar_cache_vs_full':cache_error,'npu_vs_standard_q8':npu_error,
            'npu_mean_kl_vs_standard_q8':kl,
            'custom_argmax_matches_standard':int((custom_logits.argmax(-1)==standard_logits.argmax(-1)).sum()),
            'npu_argmax_matches_standard':int((npu.argmax(-1)==standard_logits.argmax(-1)).sum()),
            'gate':{'max_relative_l2':1e-4,'passed':passed,'scope':'oracle and standard cache consistency only; NPU quality has separate gates'}}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('config','layer_errors')},indent=2))
    raise SystemExit(0 if passed else 3)


if __name__=='__main__':main()
