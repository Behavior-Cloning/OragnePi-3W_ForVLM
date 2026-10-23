import pathlib,time,json,os,resource
import torch
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
from PIL import Image
P=pathlib.Path(__file__).parent;torch.set_num_threads(2);torch.set_num_interop_threads(1)
log=open(P/'pali-wsl-caption-reference.jsonl','w')
def emit(d):log.write(json.dumps(d,ensure_ascii=False)+'\n');log.flush();print(d,flush=True)
t=time.monotonic();processor=AutoProcessor.from_pretrained(P/'model',local_files_only=True);model=PaliGemmaForConditionalGeneration.from_pretrained(P/'model',torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,local_files_only=True).eval();emit(dict(event='loaded',load_s=time.monotonic()-t,parameters=sum(x.numel() for x in model.parameters()),rss_peak_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
img=Image.open(P/'gemma.cpp-main/paligemma/testdata/image.ppm').convert('RGB')
for prompt in ['Describe this image in one sentence.']:
 inp=processor(images=img,text=prompt,return_tensors='pt');inp={k:v.to(torch.bfloat16) if v.is_floating_point() else v for k,v in inp.items()};t=time.monotonic()
 with torch.inference_mode():out=model.generate(**inp,max_new_tokens=32,do_sample=False)
 emit(dict(event='output',prompt=prompt,input_tokens=inp['input_ids'].shape[1],output_ids=out[0,inp['input_ids'].shape[1]:].tolist(),text=processor.decode(out[0,inp['input_ids'].shape[1]:],skip_special_tokens=True),wall_s=time.monotonic()-t,rss_peak_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
emit(dict(event='complete'))
