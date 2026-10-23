import pathlib,json,hashlib,numpy as np
D=pathlib.Path(__file__).parent
R=D.parents[3]/'llama32-full-npu-20261023'
# Resolve the owner directory from frozen provenance instead of path-depth assumptions.
inputs={s:json.loads((D/(s+'-inputs.json')).read_text()) for s in ['q4','q8']}
R=pathlib.Path(inputs['q4']['freeze_provenance']['source_inputs']).parents[1]
owner=pathlib.Path('/mnt/f/Files/OPi-Lab/repos/OragnePi-3W_ForVLM/research/llama32-full-npu-20261023')
refs={s:json.loads((D/s/'results/reference.json').read_text()) for s in ['q4','q8']}
old=json.loads((owner/'reference-q4.json').read_text());rows=[]
for i,case in enumerate(inputs['q4']['cases']):
 row={'id':case['id'],'prompt':case['text'],'input_tokens':len(case['tokens'])}
 for s in ['q4','q8']:
  c=refs[s]['cases'][i];path=D/s/'results'/(c['id']+'-logits.f32');data=np.memmap(path,dtype='<f4',mode='r').reshape(-1,128256)
  assert data.shape[0]==len(c['tokens']);assert hashlib.sha256(path.read_bytes()).hexdigest()==c['logits_sha256']
  ended=c['tokens'][-1] in inputs[s]['eos'];row[s]=dict(tokens=len(c['tokens']),content_tokens=len(c['tokens'])-int(ended),stopped_by='EOS' if ended else 'max_new_96',last_token=c['tokens'][-1],first_token=c['tokens'][0],text=c['text'],logits_sha256=c['logits_sha256'],logits_rows=data.shape[0],logits_columns=data.shape[1])
  if s=='q4':a=np.array(data[0],dtype=np.float64)
  else:b=np.array(data[0],dtype=np.float64)
 # Only first-step logits have identical contexts; later free-generation contexts differ.
 pa=np.exp(a-a.max());pa/=pa.sum();pb=np.exp(b-b.max());pb/=pb.sum()
 row['same_prompt_first_step_only']=dict(logit_centered_rmse=float(np.sqrt(np.mean(((a-a.mean())-(b-b.mean()))**2))),q4_top1=int(a.argmax()),q8_top1=int(b.argmax()),top10_overlap=len(set(np.argsort(a)[-10:])&set(np.argsort(b)[-10:])),q4_top1_probability_in_q4=float(pa[a.argmax()]),q8_top1_probability_in_q8=float(pb[b.argmax()]),KL_q8_to_q4_nats=float(np.sum(pb*(np.log(pb+1e-300)-np.log(pa+1e-300)))))
 prior=next(c for c in old['cases'] if c['id']==case['id']);row['q4_original_32step_prefix_exact']=refs['q4']['cases'][i]['tokens'][:len(prior['tokens'])]==prior['tokens']
 rows.append(row)
resources={}
for s in ['q4','q8']:
 t=json.loads((D/s/'telemetry.json').read_text());resources[s]=dict(max_RSS_bytes=max(x.get('VmRSS',0) for x in t),min_system_available_bytes=min(x['available_bytes'] for x in t),max_process_swap_bytes=max(x.get('VmSwap',0) for x in t),run_status=json.loads((D/s/'run-status.json').read_text()))
out=dict(scope='Host FP32 arithmetic from quantized GGUF, no NPU or board performance claims; 3 examples only',source_hashes={s:refs[s]['source_sha256'] for s in refs},same_prompt_ids=all(x['tokens']==y['tokens'] for x,y in zip(inputs['q4']['cases'],inputs['q8']['cases'])),max_new=96,greedy=True,all16_layers=True,full_vocab=128256,first_step_top1_changed=sum(r['q4']['first_token']!=r['q8']['first_token'] for r in rows),resources=resources,cases=rows)
(D/'comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(out,ensure_ascii=False,indent=2))
