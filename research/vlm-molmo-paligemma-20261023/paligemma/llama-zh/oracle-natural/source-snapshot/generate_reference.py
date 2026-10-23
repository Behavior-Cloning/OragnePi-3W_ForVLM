"""Full16-layer FP32 oracle from selected GGUF. Saves complete-vocabulary logits."""
from model import *
import argparse,time
p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--max-new',type=int,default=32);a=p.parse_args()
if not a.execute:print('Dry run: real 16-layer CPU reference, not NPU performance; explicit --execute required.');raise SystemExit()
torch.set_num_threads(2);spec=json.loads((BUILD/'inputs.json').read_text());reader,tensors=load();layers=[decoder(reader,i) for i in range(16)];heads=[HeadShard(tensors,i*32064,(i+1)*32064).eval() for i in range(4)];out=BUILD/'oracle-natural';out.mkdir(exist_ok=True);rows=[];cap=spec['capacity'];tok=build_tokenizer(reader)
with torch.inference_mode():
 for case in spec['cases']:
  ids=case['tokens'];n=len(ids);cache=torch.zeros(16,2,1,8,cap,64);hidden=torch.from_numpy(deq(tensors['token_embd.weight'],ids)[None]);start=time.monotonic()
  for i,l in enumerate(layers):
   q,k,v=l.project(hidden,*l.rope(torch.arange(n)));cache[i,0,:,:,:n]=k;cache[i,1,:,:,:n]=v;hidden=l.full(hidden)
  h=hidden[:,-1:];generated=[];times=[];logit_path=out/(case['id']+'-logits.f32')
  with logit_path.open('wb') as f:
   for step in range(a.max_new):
    logits=torch.cat([head(h) for head in heads],-1)[0,0].numpy();assert logits.shape==(128256,);assert np.isfinite(logits).all();logits.astype('<f4').tofile(f);token=int(logits.argmax());generated.append(token);times.append(time.monotonic()-start)
    if token in spec['eos']:break
    pos=n+step;assert pos<cap;h=torch.from_numpy(deq(tensors['token_embd.weight'],[token])[None]);mask=torch.full((1,1,1,cap+1),-10000.);mask[...,:pos]=0;mask[...,-1]=0
    for i,l in enumerate(layers):
     h,k,v=l(h,cache[i,0],cache[i,1],mask,*l.rope(torch.tensor([pos])));cache[i,0,:,:,pos:pos+1]=k;cache[i,1,:,:,pos:pos+1]=v
  row=dict(id=case['id'],prompt_tokens=n,tokens=generated,text=tok.decode(generated,skip_special_tokens=True),elapsed_cpu_seconds=time.monotonic()-start,logits=str(logit_path),logits_sha256=sha(logit_path),note='Host FP32 dequantized-source oracle, not NPU TTFT or native CPU benchmark');rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
result=dict(source=str(MODEL),source_sha256=sha(MODEL),source_precision=MODEL.name+' dequantized FP32; not BF16',cases=rows,all16_layers=True,full_vocab=128256)
result.update(max_new=a.max_new,inputs_sha256=sha(BUILD/'inputs.json'),torch_threads=torch.get_num_threads());(out/'reference.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
