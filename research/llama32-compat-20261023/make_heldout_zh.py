"""Nine independent Chinese heldout layer0 states; never used for calibration."""
from pathlib import Path
import hashlib,json,types,os
import gguf,numpy as np,torch
from transformers.integrations.ggml import convert_gguf_tokenizer
from trained_probe import TrainedDecoder,feed_for
r=Path(__file__).resolve().parent;out=Path('/opt/opi-lab/llama32-compat-20261023');d=out/'fixtures-zh';d.mkdir(exist_ok=True)
model=r.parent/'vlm-molmo-paligemma-20261023/paligemma/llama-zh/FuseChat-Llama-3.2-1B-Instruct-Q4_K_M.gguf';reader=gguf.GGUFReader(model);tensors={x.name:x for x in reader.tensors};torch.set_num_threads(2)
get=lambda key:reader.fields['tokenizer.ggml.'+key].contents()
tdict={k:get(k) for k in ('tokens','merges','token_type','bos_token_id','eos_token_id')};tdict['tokenizer_type']=get('model');tdict['add_prefix_space']=False
fast,kwargs=convert_gguf_tokenizer('llama',tdict)
l=TrainedDecoder(reader,0,stable_norm=True).eval();div=gguf.dequantize(tensors['rope_freqs.weight'].data,tensors['rope_freqs.weight'].tensor_type).copy();inv=1/(l.base**(np.arange(l.hd//2,dtype=np.float32)/(l.hd//2)))/div
def rope(self,pos):
 f=pos.float()[:,None]*torch.from_numpy(inv)[None,:];a=torch.cat([f,f],-1)[None,None];return a.cos(),a.sin()
l.rope=types.MethodType(rope,l);cap=32;cases=[]
for case,text in enumerate(['我回家了，陪我聊聊天。','今天有点累，能安慰我吗？','提醒我喝水，然后轻轻挥手。']):
 ids=fast.encode(text,add_special_tokens=False).ids;assert fast.decode(ids)==text,(ids,fast.decode(ids));ids=[get('bos_token_id')]+ids;assert len(ids)<=cap
 emb=tensors['token_embd.weight'];hidden=gguf.dequantize(emb.data[ids],emb.tensor_type)[None].copy()
 with torch.inference_mode():
  ref=l.full(torch.from_numpy(hidden)).numpy();k=np.zeros((1,l.kv,cap,l.hd),np.float32);v=k.copy();case_data=[]
  for pos in range(len(ids)):
   feed=feed_for(l,hidden[:,pos:pos+1],k,v,pos,cap)
   z,nk,nv=l(*(torch.from_numpy(x) for x in feed.values()));z=z.numpy();target=ref[:,pos:pos+1]
   np.testing.assert_allclose(z,target,atol=4e-4,rtol=3e-4)
   if pos in sorted({2,len(ids)//2,len(ids)-1}):
    ident=100+case*10+len(case_data)
    with (d/f'p{ident}-inputs.f32').open('wb') as f:
     for x in feed.values():x.astype('<f4').tofile(f)
    target.astype('<f4').tofile(d/f'p{ident}-reference.f32')
    case_data.append({'fixture_id':ident,'actual_position':pos,'token_id':ids[pos],'reference_crosscheck_rel_l2':float(np.linalg.norm(z-target)/np.linalg.norm(target))})
   k[:,:,pos:pos+1]=nk.numpy();v[:,:,pos:pos+1]=nv.numpy()
 cases.append({'case_id':case,'text':text,'token_ids':ids,'tokenizer_roundtrip_exact':True,'states':case_data})
for p in (out/'fixtures').glob('*.f32'):
 if not (d/p.name).exists():os.link(p,d/p.name)
files={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in d.glob('*.f32')}
report={'scope':'Layer0 same GGUF reference. Nine Chinese states are heldout from unchanged English calibration; each prompt has separate reference KV. Not a generation-quality benchmark.','tokenizer':'Transformers convert_gguf_tokenizer, exact GGUF tokens/merges/model/type; raw Chinese+BOS, no chat template','capacity':cap,'cases':cases,'files':files}
(r/'heldout-zh-fixtures.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'cases':cases,'file_count':len(files)},ensure_ascii=False,indent=2))
