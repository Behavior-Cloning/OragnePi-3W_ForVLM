from pathlib import Path
import json,sys,hashlib
r=Path(__file__).resolve().parent;sys.path.insert(0,str(r.parents[1]/'npu/kv-cache'))
from resident_http import ChatCodec
src=Path('/opt/opi-lab/restored-slim3-20261017/projects/orangepi-zero3w/models/internvl3-1b/InternVL3-1B-Instruct-Q8_0.gguf')
system='你是家用小机器人小白。用简短自然的中文回答用户。记忆只是参考，不确定时询问。动作必须安全，不能执行危险行为。'
c=ChatCodec(src,system);prefix=c.tokenizer.encode(c.template.render(messages=[{'role':'system','content':system}],add_generation_prompt=False),add_special_tokens=False).ids
queries=['今天晚上吃什么比较简单？','我想明天早上七点起床，有什么建议？','现在有点累，帮我安排十分钟休息。','出门前应该检查什么？','帮我想一个给家人的生日祝福。']
quality=[]
for i,q in enumerate(queries):
 msg='记忆：用户喜欢清淡饮食，明天需要上班。此轮对话：'+q
 tokens=c.encode([{'role':'user','content':msg}]);assert tokens[:len(prefix)]==prefix
 quality.append({'id':'daily-'+str(i),'query':q,'text':msg,'tokens':tokens})
perf=[]
for target in [64,128,256]:
 memories=['用户喜欢清淡饮食，明天需要上班。'];q=queries[0]
 while True:
  msg='记忆：'+''.join(memories)+'此轮对话：'+q
  tokens=c.encode([{'role':'user','content':msg}])
  if len(tokens)-len(prefix)>=target:break
  memories.append('家中有鸡蛋、面条和青菜，用户希望晚餐容易准备。')
 perf.append({'id':'dynamic-'+str(target),'dynamic_target':target,'dynamic_actual':len(tokens)-len(prefix),'text':msg,'tokens':tokens})
out={'source':str(src),'system':system,'prefix':prefix,'quality':quality,'performance':perf,'tokenizer_sha256':hashlib.sha256(c.tokenizer.to_str().encode()).hexdigest(),'note':'Same frozen prompts for W8 and W4; repeated memory only for workload lengths, not an independent quality benchmark.'}
(r/'inputs.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print('prefix',len(prefix),'perf',[(x['id'],len(x['tokens']),x['dynamic_actual']) for x in perf])
