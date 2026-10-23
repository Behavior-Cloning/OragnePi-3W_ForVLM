"""Future sequential FP32 calibration pass. No NPU compilation or board access."""
from model import *
import argparse,gc,shutil
p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
if not a.execute:print('Dry run: prepares each of 16 real layer inputs, then head inputs and embedding; pass --execute after compact.');raise SystemExit()
assert shutil.disk_usage('/mnt/f').free>8*1024**3
torch.set_num_threads(2);reader,tensors=load();tok=build_tokenizer(reader);spec=json.loads((BUILD/'inputs.json').read_text());cap=spec['capacity'];cal=BUILD/'calibration';cal.mkdir(parents=True,exist_ok=True)
texts=['请简单介绍你自己。','我想听一个温暖的小故事。','现在不要说话，等我叫你。'];states=[]
for i,text in enumerate(texts):
 ids=tok.apply_chat_template([dict(role='system',content=spec['system']),dict(role='user',content=text)],tokenize=True,add_generation_prompt=True);states.append(dict(id=i,text=text,tokens=ids,hidden=deq(tensors['token_embd.weight'],ids)[None]))
for li in range(16):
 layer=decoder(reader,li);d=cal/f'layer{li}';d.mkdir(exist_ok=True);entries=[];meta=[]
 for case in states:
  x=torch.from_numpy(case['hidden']);n=x.shape[1]
  with torch.inference_mode():
   q,k,v=layer.project(x,*layer.rope(torch.arange(n)));y=layer.full(x)
  for pos in sorted(set([0,1,len(spec['prefix'])-1,len(spec['prefix'])+1,n-1])):
   pk=np.zeros((1,8,cap,64),np.float32);pv=np.zeros_like(pk);pk[:,:,:pos]=k[:,:,:pos].numpy();pv[:,:,:pos]=v[:,:,:pos].numpy();mask=np.full((1,1,1,cap+1),-10000,np.float32);mask[...,:pos]=0;mask[...,-1]=0;cos,sin=layer.rope(torch.tensor([pos]));values=[x[:,pos:pos+1].numpy(),pk,pv,mask,cos.numpy(),sin.numpy()];names=[]
   for name,value in zip(INPUTS,values):
    f=f'c{case["id"]}_p{pos}_{name}.npy';np.save(d/f,value);names.append(f)
   np.save(d/f'c{case["id"]}_p{pos}_reference.npy',y[:,pos:pos+1].numpy());entries.append(' '.join(names));meta.append(dict(case=case['id'],pos=pos))
  case['hidden']=y.numpy()
 (d/'dataset.txt').write_text('\n'.join(entries)+'\n');(d/'meta.json').write_text(json.dumps(meta,indent=2));print('calibrated real input layer',li,flush=True);del layer;gc.collect()
hd=cal/'head';hd.mkdir(exist_ok=True);entries=[]
for case in states:
 for pos in [0,len(case['tokens'])-1]:
  name=f'c{case["id"]}_p{pos}_hidden.npy';np.save(hd/name,case['hidden'][:,pos:pos+1]);entries.append(name)
(hd/'dataset.txt').write_text('\n'.join(entries)+'\n')
with (BUILD/'embeddings.f32').open('wb') as f:
 for start in range(0,128256,1024):f.write(deq(tensors['token_embd.weight'],slice(start,min(start+1024,128256))).astype('<f4').tobytes())
manifest=dict(source_sha256=sha(MODEL),capacity=cap,calibration_texts=texts,quality_texts=[x['text'] for x in spec['cases']],layers=16,head_shards=4,embedding_sha256=sha(BUILD/'embeddings.f32'),embedding_bytes=(BUILD/'embeddings.f32').stat().st_size,source_precision=MODEL.name+' dequantized FP32; not BF16',note='Each layer receives its own preceding FP32 layers output. No reuse of layer0 activation calibration.')
(BUILD/'calibration-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));(ROOT/('calibration-manifest-'+SOURCE_TAG+'.json')).write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print('complete',flush=True)
