from pathlib import Path
import json,sys
r=Path("/mnt/f/Files/OPi-Lab/repos/OragnePi-3W_ForVLM");sys.path.insert(0,str(r/'npu/kv-cache'))
from resident_http import ChatCodec
base=Path('/opt/opi-lab/restored-slim3-20261017/projects/orangepi-zero3w/models')
models={'smol500':base/'smolvlm-500m/SmolVLM-500M-Instruct-Q8_0.gguf','intern1b':base/'internvl3-1b/InternVL3-1B-Instruct-Q8_0.gguf'}
system='你是家用小机器人小白。用简短自然的中文回答用户。记忆只是参考，不确定时询问。动作必须安全，不能执行危险行为。'
queries=['今天晚上吃什么比较简单？','我想明天早上七点起床，有什么建议？','现在有点累，帮我安排十分钟休息。','出门前应该检查什么？','帮我想一个给家人的生日祝福。']
out={}
for name,path in models.items():
 c=ChatCodec(path,system);messages=[{'role':'system','content':system}]
 if c.typed_content:messages=[{'role':'system','content':[{'type':'text','text':system}]}]
 pre=c.tokenizer.encode(c.template.render(messages=messages,add_generation_prompt=False),add_special_tokens=False).ids
 prompts=[c.encode([{'role':'user','content':'记忆：用户喜欢清淡饮食，明天需要上班。此轮对话：'+q}]) for q in queries]
 assert all(t[:len(pre)]==pre for t in prompts)
 out[name]={'source':str(path),'system':system,'prefix':pre,'prompts':prompts,'queries':queries,'prefix_tokens':len(pre),'prompt_tokens':[len(t) for t in prompts]}
(r/'research/jev-validation/resident-inputs.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print({k:{'prefix':v['prefix_tokens'],'prompts':v['prompt_tokens']} for k,v in out.items()})
