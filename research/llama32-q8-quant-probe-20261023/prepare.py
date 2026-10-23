"""Q8 source layer0 and full decoder head reference, separate Chinese cal/holdout."""
from model import *
import gc, shutil, argparse
p=argparse.ArgumentParser();p.add_argument("--layer0-only",action="store_true");a=p.parse_args()
RECORD=Path(os.environ.get("PROBE_RECORD_ROOT",str(ROOT)));RECORD.mkdir(parents=True,exist_ok=True)
assert shutil.disk_usage('/mnt/f').free > 8*1024**3
assert sha(MODEL)==('c7db2b30d2b176998db90304d1ac6dd259379bd75f9ea8b6a094be88ea47b1b4' if 'Q8' in MODEL.name else '8bcd46580c2eb069fde82b4c16f8f2981901d653fea1340ec560b6097f08f97a')
torch.set_num_threads(2); reader,tensors=load(); tok=build_tokenizer(reader); BUILD.mkdir(exist_ok=True,parents=True)
system='你是家庭陪伴机器人，请用简短中文回答。'
cal=['请介绍一下你能怎样陪伴我。','周末下午我们一起听轻音乐吧。','现在保持安静，等我说话再回应。','我今天很开心，想和你分享好消息。']
held=['你好，我今天有点累。','明早七点提醒我起床。','不要开灯，请陪我聊两句。','妈妈还在休息，请轻声说一句早安。']
states=[]; meta=[]; layer0=BUILD/'calibration/layer0';layer0.mkdir(parents=True,exist_ok=True);fixtures=BUILD/'fixtures';fixtures.mkdir(exist_ok=True); rows=[]
for i,text in enumerate(cal+held):
 ids=tok.apply_chat_template([dict(role='system',content=system),dict(role='user',content=text)],tokenize=True,add_generation_prompt=True)
 states.append(dict(case_id=i,split='cal' if i<4 else 'heldout',text=text,tokens=ids,hidden=deq(tensors['token_embd.weight'],ids)[None]))
cap=256
for li in range(1 if a.layer0_only else 16):
 layer=decoder(reader,li)
 for case in states:
  x=torch.from_numpy(case['hidden']);n=x.shape[1]
  with torch.inference_mode():
   y=layer.full(x)
   if li==0:q,k,v=layer.project(x,*layer.rope(torch.arange(n)))
  if li==0:
   for pos in [0,1,21,27,n-1]:
    pk=np.zeros((1,8,cap,64),np.float32);pv=np.zeros_like(pk);pk[:,:,:pos]=k[:,:,:pos].numpy();pv[:,:,:pos]=v[:,:,:pos].numpy();mask=np.full((1,1,1,cap+1),-10000,np.float32);mask[...,:pos]=0;mask[...,-1]=0;co,si=layer.rope(torch.tensor([pos]));values=[x[:,pos:pos+1].numpy(),pk,pv,mask,co.numpy(),si.numpy()]
    with torch.inference_mode(): actual=layer(*[torch.from_numpy(a) for a in values])
    rel=float(np.linalg.norm(actual[0].numpy()-y[:,pos:pos+1].numpy())/np.linalg.norm(y[:,pos:pos+1].numpy()));assert rel<3e-5
    fid=len(meta);names=[]
    with (fixtures/f'p{fid}-inputs.f32').open('wb') as f:
     for name,value in zip(INPUTS,values):
      value.astype('<f4').tofile(f);nm=f'c{case["case_id"]}_p{pos}_{name}.npy';np.save(layer0/nm,value);names.append(nm)
    for oi,z in enumerate(actual):z.numpy().astype('<f4').tofile(fixtures/f'p{fid}-ref{oi}.f32')
    if case['split']=='cal':rows.append(' '.join(names))
    meta.append(dict(fixture_id=fid,case_id=case['case_id'],split=case['split'],position=pos,text=case['text'],full_vs_incremental_rel_l2=rel))
  case['hidden']=y.numpy()
 print('FP32 preceding layer',li,flush=True);del layer;gc.collect()
(layer0/'dataset.txt').write_text('\n'.join(rows)+'\n')
hmeta=[]
if not a.layer0_only:
 head=BUILD/'calibration/head';head.mkdir(exist_ok=True);headrows=[];hmeta=[]
 for case in states:
  for pos in [27,len(case['tokens'])-1]:
   x=case['hidden'][:,pos:pos+1];nm=f'c{case["case_id"]}_p{pos}_hidden.npy';np.save(head/nm,x);hid=len(hmeta);x.astype('<f4').tofile(fixtures/f'h{hid}-inputs.f32')
   if case['split']=='cal':headrows.append(nm)
   hmeta.append(dict(fixture_id=hid,case_id=case['case_id'],split=case['split'],position=pos,npy=nm))
 (head/'dataset.txt').write_text('\n'.join(headrows)+'\n')
 full=[]
 for j in range(4):
  m=HeadShard(tensors,j*32064,(j+1)*32064).eval();vals=[]
  for rec in hmeta:
   with torch.inference_mode():z=m(torch.from_numpy(np.load(head/rec['npy']))).numpy()
   z.astype('<f4').tofile(fixtures/f'h{rec["fixture_id"]}-s{j}-reference.f32');vals.append(z)
  full.append(np.concatenate(vals,axis=0));del m;gc.collect()
 all_logits=np.concatenate(full,axis=-1)
 for rec,z in zip(hmeta,all_logits):rec['full_vocabulary_argmax']=int(z.argmax());rec['top10']=np.argsort(-z.reshape(-1))[:10].tolist()
cal_hashes=set()
for rec in meta:
 rec['input_sha256']=sha(fixtures/f'p{rec["fixture_id"]}-inputs.f32')
 if rec['split']=='cal':cal_hashes.add(rec['input_sha256'])
for rec in meta:rec['seen_in_calibration']=rec['input_sha256'] in cal_hashes
manifest=dict(source=str(MODEL),source_sha256=sha(MODEL),source_precision=MODEL.name+' dequantized FP32, not original BF16',capacity=cap,system=system,calibration_texts=cal,heldout_texts=held,decoder_fixtures=meta,head_fixtures=hmeta,head_hidden_source=('Not prepared: layer0-only control' if a.layer0_only else 'Actual all16 FP32 decoder outputs from source GGUF weights; no NPU accumulated error yet'),source_tokens=[dict(case_id=s['case_id'],split=s['split'],text=s['text'],tokens=s['tokens']) for s in states])
manifest['decoder_unique_counts']={s:len(set(x['input_sha256'] for x in meta if x['split']==s)) for s in ['cal','heldout']}
manifest['decoder_heldout_unseen_count']=sum(x['split']=='heldout' and not x['seen_in_calibration'] for x in meta)
manifest['split_note']='Same fixed system prefix shares BOS/system states; unseen input hashes are independent of calibration.'
(RECORD/'fixtures-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));(BUILD/'fixtures-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print('PREPARED',flush=True)
