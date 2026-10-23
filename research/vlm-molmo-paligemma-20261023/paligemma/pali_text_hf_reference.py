import pathlib,time,json,resource,torch
from transformers import AutoTokenizer,PaliGemmaForConditionalGeneration
P=pathlib.Path(__file__).parent;torch.set_num_threads(2);torch.set_num_interop_threads(1)
f=open(P/'pali-text-hf-reference.jsonl','w')
def emit(x):f.write(json.dumps(x,ensure_ascii=False)+'\n');f.flush();print(x,flush=True)
tok=AutoTokenizer.from_pretrained(P/'model',local_files_only=True);model=PaliGemmaForConditionalGeneration.from_pretrained(P/'model',torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,local_files_only=True).eval()
for c in json.loads((P/'text-only-cases.json').read_text()):
 inp=tok(c['prompt']+'\n',return_tensors='pt',add_special_tokens=False);inp['input_ids']=torch.cat([torch.full((1,1),tok.bos_token_id,dtype=torch.long),inp['input_ids']],dim=1);inp['attention_mask']=torch.ones_like(inp['input_ids']);t=time.monotonic()
 with torch.inference_mode():out=model.generate(**inp,max_new_tokens=16,do_sample=False)
 ids=out[0,inp['input_ids'].shape[1]:].tolist();emit(dict(case=c['id'],input_ids=inp['input_ids'][0].tolist(),output_ids=ids,text=tok.decode(ids,skip_special_tokens=True),wall_s=time.monotonic()-t))
emit(dict(event='complete'))
