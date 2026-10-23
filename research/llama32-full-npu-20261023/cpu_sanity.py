"""Small real-weight CPU check; does not export/compile or access board."""
from model import *
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.nn import functional as F
torch.set_num_threads(2);reader,tensors=load();cfg=LlamaConfig(**json.loads((SOURCE/'original-config.json').read_text()));cfg._attn_implementation='eager'
inputs=json.loads((ROOT/'inputs.json').read_text());ids=inputs['cases'][0]['tokens'][-9:];x=torch.from_numpy(deq(tensors['token_embd.weight'],ids)[None]);l=decoder(reader,0)
ref=LlamaDecoderLayer(cfg,0).eval();sd={}
for n,k in [('q','self_attn.q_proj.weight'),('k','self_attn.k_proj.weight'),('v','self_attn.v_proj.weight'),('o','self_attn.o_proj.weight'),('gate','mlp.gate_proj.weight'),('up','mlp.up_proj.weight'),('down','mlp.down_proj.weight'),('norm1','input_layernorm.weight'),('norm2','post_attention_layernorm.weight')]:sd[k]=getattr(l,n)
ref.load_state_dict(sd,strict=True,assign=True)
cos,sin=l.rope(torch.arange(len(ids)));mask=torch.triu(torch.full((1,1,len(ids),len(ids)),-10000.),diagonal=1)
with torch.inference_mode():
 z=l.full(x);y=ref(x,attention_mask=mask,position_embeddings=(cos[:,0],sin[:,0]));y=y[0] if isinstance(y,tuple) else y
 error=float((z-y).abs().max());rel=float(torch.linalg.vector_norm(z-y)/torch.linalg.vector_norm(y));assert rel<2e-5
 j=HeadShard(tensors,0,32064);h=z[:,-1:];a=j(h);scale=deq(tensors['output_norm.weight']);manual=F.linear(h*torch.rsqrt((h*h).mean(-1,keepdim=True)+1e-5)*torch.from_numpy(scale),j.weight);hr=float(torch.linalg.vector_norm(a-manual)/torch.linalg.vector_norm(manual));assert hr<2e-5
report=dict(layer=0,tokens=ids,official_llama_decoder_rel_l2=rel,official_max_abs=error,head_shard0_rel_l2=hr,head_vocab_range=[0,32064],source='dequantized Q4 GGUF, not BF16',npu_tested=False)
(ROOT/'cpu-sanity.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
