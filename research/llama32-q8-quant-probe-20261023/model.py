"""FuseChat Q8-source probe arithmetic. This is not a BF16 reference."""
from pathlib import Path
import json,hashlib,types,os
import numpy as np,torch,gguf
from torch import nn
from torch.nn import functional as F
from trained_probe import TrainedDecoder,rms_norm
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'vlm-molmo-paligemma-20261023/paligemma/llama-zh'
MODEL=Path(os.environ.get('LLAMA_GGUF_MODEL',str(SOURCE/'q8-source/FuseChat-Llama-3.2-1B-Instruct-Q8_0.gguf')))
BUILD=Path(os.environ.get('LLAMA_BUILD_ROOT','/opt/opi-lab/llama32-q8-quant-probe-20261023'))
SOURCE_TAG='q8-source' if 'Q8' in MODEL.name else 'q4-source'
EVIDENCE=Path(os.environ.get('LLAMA_EVIDENCE_ROOT',str(ROOT/'build-evidence'/SOURCE_TAG)))
INPUTS=['hidden','past_k','past_v','mask','rope_cos','rope_sin'];OUTPUTS=['hidden_out','new_k','new_v']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def load():
 reader=gguf.GGUFReader(MODEL);tensors={t.name:t for t in reader.tensors};return reader,tensors
def deq(t,rows=None):return gguf.dequantize(t.data if rows is None else t.data[rows],t.tensor_type).copy()
def inverse_freq(reader):
 t=next(t for t in reader.tensors if t.name=='rope_freqs.weight');return 500000.**(-np.arange(32,dtype=np.float32)/32)/deq(t)
def decoder(reader,index):
 l=TrainedDecoder(reader,index,stable_norm=True).eval();inv=torch.from_numpy(inverse_freq(reader))
 def rope(self,positions):
  f=positions.float()[:,None]*inv[None,:];a=torch.cat([f,f],-1)[None,None];return a.cos(),a.sin()
 l.rope=types.MethodType(rope,l);return l
class HeadShard(nn.Module):
 def __init__(self,tensors,start=0,end=128256):
  super().__init__();t=tensors.get('output.weight',tensors['token_embd.weight']);self.register_buffer('weight',torch.from_numpy(deq(t,slice(start,end))));self.register_buffer('scale',torch.from_numpy(deq(tensors['output_norm.weight'])))
 def forward(self,x):return F.linear(rms_norm(x,self.scale,1e-5,True,3),self.weight)
def build_tokenizer(reader):
 from tokenizers import Tokenizer,models,pre_tokenizers,decoders,Regex,AddedToken
 from transformers import PreTrainedTokenizerFast
 f=reader.fields;vocab=f['tokenizer.ggml.tokens'].contents();merges=[tuple(m.split(' ')) for m in f['tokenizer.ggml.merges'].contents()]
 b=Tokenizer(models.BPE(vocab={v:i for i,v in enumerate(vocab)},merges=merges,byte_fallback=False))
 pattern=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
 b.pre_tokenizer=pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(pattern),behavior='isolated'),pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=False)])
 b.decoder=decoders.ByteLevel();types=f['tokenizer.ggml.token_type'].contents();b.add_special_tokens([AddedToken(vocab[i],special=True,normalized=False) for i,t in enumerate(types) if t==3])
 config=json.loads((SOURCE/'original-tokenizer_config.json').read_text())
 return PreTrainedTokenizerFast(tokenizer_object=b,bos_token='<|begin_of_text|>',eos_token='<|eot_id|>',pad_token='<|finetune_right_pad_id|>',chat_template=config['chat_template'],clean_up_tokenization_spaces=False)
