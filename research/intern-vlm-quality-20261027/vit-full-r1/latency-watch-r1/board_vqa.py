from pathlib import Path
import os,sys,json,hashlib,time,traceback
R=Path(__file__).resolve().parent
sys.path[:0]=['/opt/opi-vlm/resident-agent-20261019/deps','/opt/opi-vlm/audio-lab-20261018/deps']
from tokenizers import Tokenizer
from resident_worker_visual import ResidentWorker
cfg=json.loads((R/'parent-config.json').read_text());cases=json.loads((R/'vqa-cases.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
tokpath=Path('/opt/opi-vlm/resident-agent-20261019/tokenizer.json');assert sha(tokpath)=='97030cfe860ae6e146e62b7c84f46789f6a2a2c6edb6a32f4cb089f2bc7456e4'
tok=Tokenizer.from_file(str(tokpath));enc=lambda s:tok.encode(s,add_special_tokens=False).ids
out=R/'vqa-results';out.mkdir(exist_ok=False);dump=out/'dumps';dump.mkdir();events=(out/'events.jsonl').open('x',buffering=1)
def emit(x):
 x['monotonic']=time.monotonic();events.write(json.dumps(x,ensure_ascii=False)+'\n')
sysbody=cases['system'];st=enc('<|im_start|>system\n'+sysbody+'<|im_end|>\n');pre=enc('<|im_start|>system\n'+sysbody+'<|im_end|>\n<|im_start|>user\n<img>')
imageid=tok.token_to_id('<IMG_CONTEXT>');assert imageid is not None
inputs=[]
for c in cases['cases']:
 tail=enc('</img>\n'+c['question']+'<|im_end|>\n<|im_start|>assistant\n');ids=pre+[imageid]*64+tail
 inputs.append({**c,'tokens':ids,'image_start':len(pre),'system_body':len(enc(sysbody)),'system_template':len(st)-len(enc(sysbody)),'system_total':len(st),'dynamic_image':64,'dynamic_text':len(enc(c['question'])),'template_boundaries':len(ids)-len(st)-64-len(enc(c['question']))})
 assert 32<=len(st)<=64 and 64+len(enc(c['question']))<=128
(R/'vqa-frozen-inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2)+'\n')
graph=cfg['parent_graph'];env=['OPI_GROUP_PREFILL=1','OPI_MASK_TEMPLATE=1','OPI_TTFT_TAIL_MIN=8','OPI_GREEDY_I16_SINGLE_PASS=1','OPI_PREFILL8_DIR='+cfg['parent_tail'],'OPI_COMPACT64_DIR='+cfg['compact_graph']]
cmd=['env',*env,'taskset','-c','7',str(R/'resident-visual'),graph,'24','448','151674',graph+'/embeddings.f32','32','151645']
assert sha(R/'resident-visual')=='8e5381670c1cd724ac1e08e713d0c2aae83cc4a71b2eeaf9766868e897bc814c'
for p,h in cfg['graphs_sha256'].items():assert sha(p)==h
w=ResidentWorker(cmd,'/opt/opi-vlm/resident-agent-20261019',str(out/'worker.log'),dump_dir=str(dump),timeout=180)
emit(w.ready);records=[]
def snapshot():
 return {'at':time.monotonic(),'meminfo':Path('/proc/meminfo').read_text(),'vmstat':Path('/proc/vmstat').read_text(),'memory_pressure':Path('/proc/pressure/memory').read_text(),'io_pressure':Path('/proc/pressure/io').read_text(),'stat':Path('/proc',str(w.process.pid),'stat').read_text(),'status':Path('/proc',str(w.process.pid),'status').read_text(),'io':Path('/proc',str(w.process.pid),'io').read_text(),'temps':{str(p):int(p.read_text()) for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')}}
def req(ids,**kw):
 before=snapshot()
 m=list(w.generate(ids,**kw))[-1]
 after=snapshot()
 emit({'event':'request_resources','id':m['id'],'before':before,'after':after})
 return m
def visual(c):
 p=R/(c['image']+'.f32');v=p.read_bytes();assert len(v)==64*896*4
 return (c['image_start'],len(st),v)
try:
 for c in inputs*3:
  req(st,max_new=1,reset=True)
  m=req(c['tokens'],max_new=32,visual=visual(c));assert m['reused_tokens']==32
  row={**c,'metrics':m,'answer':tok.decode(m['tokens'],skip_special_tokens=True),'embedding_sha256':sha(R/(c['image']+'.f32'))};emit(row);records.append(row)
 # Same text/placeholder IDs, different visual content. Each reference primes only system.
 A=inputs[0];B=inputs[3];assert A['tokens']==B['tokens']
 hashes={}
 for name,c,prime in [('B-reference',B,True),('A',A,False),('B-after-A',B,False)]:
  if prime:req(st,max_new=1,reset=True)
  m=req(c['tokens'],max_new=4,feedback=[100662]*4,diagnostic=True,visual=visual(c));assert m['reused_tokens']==32
  f=dump/(str(m['id'])+'-logits.f32');assert f.stat().st_size==2426784
  hashes[name]=sha(f);emit({'event':'cache_diagnostic','name':name,'metrics':m,'logits_sha256':hashes[name]})
 assert hashes['B-reference']==hashes['B-after-A'],'image replacement cache mismatch'
 assert hashes['A']!=hashes['B-reference'],'different images did not affect logits'
 # Returning to text must also cap stale image KV, even when IDs are identical.
 m=req(B['tokens'],max_new=4,feedback=[100662]*4,diagnostic=True);h1=sha(dump/(str(m['id'])+'-logits.f32'));assert m['reused_tokens']==32
 req(st,max_new=1,reset=True)
 m2=req(B['tokens'],max_new=4,feedback=[100662]*4,diagnostic=True);h2=sha(dump/(str(m2['id'])+'-logits.f32'));assert h1==h2
 emit({'event':'image_to_text_cache','same_logits':True,'sha256':h1})
 result={'status':'complete','scope':'board NPU language with NPU24block embeddings and offline FP32 stem/projector; not full on-board tower or co-resident end-to-end','records':records,'cache_hashes':hashes}
except BaseException:
 result={'status':'failed','error':traceback.format_exc(),'records':records};raise
finally:
 closed=w.close();emit(closed);result['closed']=closed;assert closed['driver_bytes']==0
 (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');events.close()
