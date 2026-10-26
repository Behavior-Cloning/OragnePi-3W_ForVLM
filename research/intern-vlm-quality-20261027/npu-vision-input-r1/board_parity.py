from pathlib import Path
import json,os,hashlib,struct,time,traceback
from resident_worker_visual import ResidentWorker
R=Path(__file__).resolve().parent
cfg=json.loads((R/'parent-config.json').read_text());doc=json.loads((R/'parent-inputs.json').read_text())
out=R/'parity-results';out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(cfg['binary'])==cfg['binary_sha256']
assert sha(R/'resident-visual')=='8e5381670c1cd724ac1e08e713d0c2aae83cc4a71b2eeaf9766868e897bc814c'
for p,h in cfg['graphs_sha256'].items():assert sha(p)==h,p
cases=[next(c for c in doc['performance'] if c['id']==name) for name in ['s32-d32-0','s64-d64-0','s64-d128-0']]
records=[];events=(out/'events.jsonl').open('x',buffering=1)
def emit(e):events.write(json.dumps(e)+'\n')
def req(w,tokens,**kw):return list(w.generate(tokens,**kw))[-1]
try:
 for phase in ['parent','candidate','equivalent-override']:
  dumps=out/(phase+'-dumps');dumps.mkdir()
  env=['OPI_GROUP_PREFILL=1','OPI_MASK_TEMPLATE=1','OPI_TTFT_TAIL_MIN=8','OPI_GREEDY_I16_SINGLE_PASS=1','OPI_PREFILL8_DIR='+cfg['parent_tail'],'OPI_COMPACT64_DIR='+cfg['compact_graph']]
  binary=cfg['binary'] if phase=='parent' else str(R/'resident-visual')
  graph=cfg['parent_graph'];cmd=['env',*env,'taskset','-c','7',binary,graph,'24','448','151674',graph+'/embeddings.f32','32','151645']
  w=ResidentWorker(cmd,'/opt/opi-vlm/resident-agent-20261019',str(out/(phase+'.log')),dump_dir=str(dumps),timeout=180)
  emit({'event':'ready','phase':phase,**w.ready})
  try:
   for c in cases:
    req(w,c['prefix_tokens'],max_new=1,reset=True)
    kw={}
    if phase=='equivalent-override':
     start=c['static_tokens']+3;count=min(64,len(c['tokens'])-start)
     rows=[]
     with open(graph+'/embeddings.f32','rb') as f:
      for token in c['tokens'][start:start+count]:
       f.seek(token*896*4);v=f.read(896*4);assert len(v)==896*4;rows.append(v)
     kw['visual']=(start,c['static_tokens'],b''.join(rows))
    m=req(w,c['tokens'],max_new=4,feedback=[100662]*4,diagnostic=True,**kw)
    f=dumps/(str(m['id'])+'-logits.f32');assert f.stat().st_size==2426784
    h=sha(f);row={'phase':phase,'case':c['id'],'system_total':c['static_tokens'],'dynamic_body':c['body_tokens'],'dynamic_template':8,'metrics':m,'logits_sha256':h};records.append(row);emit(row)
    assert m['reused_tokens']==c['static_tokens'] and m['new_tokens']==c['body_tokens']+8
  finally:
   closed=w.close();emit({'phase':phase,**closed});assert closed['driver_bytes']==0
 for c in cases:
  rows=[x for x in records if x['case']==c['id']]
  assert len({x['logits_sha256'] for x in rows})==1,(c['id'],'logits differ')
 result={'status':'passed','scope':'text and equivalent word-vector injection numerical parity, NOT real-image VQA','records':records}
except BaseException:
 result={'status':'failed','error':traceback.format_exc(),'records':records};raise
finally:
 (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');events.close()
