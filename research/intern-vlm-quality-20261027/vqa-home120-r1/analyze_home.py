from pathlib import Path
import json,hashlib,statistics,math,re,html
from PIL import Image
P=Path(__file__).resolve().parent
load=lambda f:json.loads((P/f).read_text())
cases=load('vqa-cases.json');grades=load('grades.json');summary={};items=[]
def stats(xs):
 xs=sorted(xs);n=len(xs)
 def pct(p):
  q=(n-1)*p;i=int(q);return xs[i]+(xs[min(i+1,n-1)]-xs[i])*(q-i)
 return {'n':n,'min':xs[0],'p50':pct(.5),'p95':pct(.95),'max':xs[-1]}
def mem(text,key):return int(re.search(r'^'+key+r':\s*(\d+)',text,re.M).group(1))
for side in [224]:
 v=load(f'vision-{side}/summary.json');a=load(f'vqa-{side}/summary.json')
 assert v['status']==a['status']=='complete';assert v['closed']['driver_bytes']==a['closed']['driver_bytes']==0
 hot=[r for r in v['records'] if not r['warmup']];assert len(hot)==len(a['records'])==120
 vm={r['id']:r for r in hot};am={r['id']:r for r in a['records']}
 for c in cases['cases']:
  assert am[c['id']]['question']==c['question'];assert am[c['id']]['embedding_sha256']==vm[c['id']]['vector_sha256']
 resources=[json.loads(l) for l in (P/f'vision-{side}/resources.jsonl').read_text().splitlines()]
 le=[json.loads(l) for l in (P/f'vqa-{side}/events.jsonl').read_text().splitlines()]
 snapshots=[x[t] for x in le if x.get('event')=='request_resources' for t in ['before','after']]
 cpu=[]
 for x,y in zip(resources,resources[1:]):
  xx=list(map(int,x['cpu'].split()[1:9]));yy=list(map(int,y['cpu'].split()[1:9]));d=[b-a for a,b in zip(xx,yy)];total=sum(d)
  if total>0:cpu.append(100*(total-d[3]-d[4])/total)
 cats={}
 for c in cases['cases']:
  g=grades[c['id']][str(side)];assert g['score'] in [0,1];cat=cats.setdefault(c['category'],{'correct':0,'n':0,'abstain':0});cat['correct']+=g['score'];cat['n']+=1;cat['abstain']+=bool(g.get('abstain'))
 metrics=[r['metrics'] for r in a['records']]
 s={'quality':cats,'correct':sum(x['correct'] for x in cats.values()),'n':120,'vision_model_s':stats([r['metrics']['seconds'] for r in hot]),'vision_api_s':stats([r['metrics']['api_seconds'] for r in hot]),'vision_wall_s':stats([r['vision_wall_s'] for r in hot]),'jpeg_to_vectors_s':stats([r['jpeg_to_vectors_s'] for r in hot]),'preprocess_s':stats([r['preprocess_s'] for r in hot]),'ttft_s':stats([m['ttft_seconds'] for m in metrics]),'decode_tps':stats([m['decode_tokens_per_second'] for m in metrics]),'prefill_including_scalar_s':stats([m['prefill_seconds']+m['scalar_prompt_seconds'] for m in metrics]),'scalar_prompt_s':stats([m['scalar_prompt_seconds'] for m in metrics]),'prefill_s':stats([m['prefill_seconds'] for m in metrics]),'language_ready':[x for x in le if x.get('event')=='ready'],'language_total_s':stats([m['total_seconds'] for m in metrics]),'segment_sum_first_s':stats([vm[r['id']]['jpeg_to_vectors_s']+r['metrics']['ttft_seconds'] for r in a['records']]),'segment_sum_complete_s':stats([vm[r['id']]['jpeg_to_vectors_s']+r['metrics']['total_seconds'] for r in a['records']]),'ready':v['ready'],'closed_vision':v['closed'],'closed_language':a['closed'],'vision_cpu_allcores_pct':stats(cpu),'vision_temp_max_c':max(t for r in resources for t in r['temps'].values())/1000,'language_temp_max_c':max(t for r in snapshots for t in r['temps'].values())/1000,'vision_memavailable_min_mib':min(mem(r['meminfo'],'MemAvailable') for r in resources)/1024,'language_memavailable_min_mib':min(mem(r['meminfo'],'MemAvailable') for r in snapshots)/1024,'language_rss_mib':stats([m['rss_kib']/1024 for m in metrics]),'language_driver_bytes':max(m['driver_bytes'] for m in metrics),'swap_free_first_kib':mem(snapshots[0]['meminfo'],'SwapFree'),'swap_free_last_kib':mem(snapshots[-1]['meminfo'],'SwapFree'),'tokens':{key:stats([r[key] for r in a['records']]) for key in ['system_body','system_template','system_total','dynamic_image','dynamic_text','template_boundaries']},'runtime_tokens':{key:stats([m[key] for m in metrics]) for key in ['prompt_tokens','reused_tokens','new_tokens','generated_tokens']}}
 firstvm=dict(line.split() for line in snapshots[0]['vmstat'].splitlines());lastvm=dict(line.split() for line in snapshots[-1]['vmstat'].splitlines())
 s['language_vmstat_delta']={k:int(lastvm[k])-int(firstvm[k]) for k in ['pswpin','pswpout','pgmajfault','oom_kill']}
 summary[str(side)]=s
 for c in cases['cases']:
  r=am[c['id']];items.append({'side':side,'id':c['id'],'answer':r['answer'],'grade':grades[c['id']][str(side)],'metrics':r['metrics']})
(P/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
(P/'scored-answers.json').write_text(json.dumps(items,ensure_ascii=False,indent=2)+'\n')
A={str(side):{r['id']:r for r in load(f'vqa-{side}/summary.json')['records']} for side in [224]}
labels={'hair':'发色','expression':'可见表情','desktop_object':'桌面物品','desktop_relation':'位置关系','indoor':'室内场景','outdoor':'室外场景','people_count':'人物数量','clothing':'衣服类别','activity':'人物活动','interaction':'可见共同活动/互动'}
E=html.escape
preview=P/'gallery-images';preview.mkdir(exist_ok=True)
sections=[]
for c in cases['cases']:
 cards=[]
 for side in [224]:
  im=Image.open(P/'images'/(c['id']+'.jpg')).convert('RGB').resize((side,side),Image.Resampling.BICUBIC);im.save(preview/(c['id']+f'-{side}.png'))
  a=A[str(side)][c['id']];g=grades[c['id']][str(side)]
  cards.append(f'<div class="card"><h3>{side}×{side} → {16 if side==112 else 64}视觉token</h3><img width="{side}" height="{side}" src="gallery-images/{c["id"]}-{side}.png"><p>{E(a["answer"])}</p><p class="{"ok" if g["score"] else "bad"}">{"正确" if g["score"] else "未通过"}：{E(g["reason"])}</p><small>语言TTFT {a["metrics"]["ttft_seconds"]*1000:.1f} ms · 输出 {a["metrics"]["generated_tokens"]} tokens</small></div>')

 sections.append(f'<section data-category="{c["category"]}"><h2>{c["id"]} · {labels[c["category"]]} · {E(c["question"])}</h2><p>参考：{E(c["reference_zh"])} · 标注：{E(c["label_provenance"])}</p><div class="pair">'+''.join(cards)+f'</div><details><summary>图片来源和溯源</summary><a href="{E(c["image_url"])}">原图</a><p>COCO image {c["image_id"]}; source question {c["source_question_id"]}; SHA {c["image_sha256"]}</p></details></section>')
page='<!doctype html><meta charset="utf-8"><title>家居社交陪伴 VQA：224 NPU W8</title><style>body{font:16px/1.6 system-ui;margin:30px auto;max-width:1100px;padding:0 20px;background:#f5f6f8;color:#18202c}section{background:white;padding:20px;margin:20px 0;border-radius:12px}.pair{display:flex;gap:30px}.card{flex:1;min-width:0}img{object-fit:contain;background:#eee}h2{font-size:19px}.ok{color:#087343}.bad{color:#b43120}small{color:#555}select{font:inherit;padding:8px}details{overflow-wrap:anywhere}@media(max-width:600px){.pair{display:block}}</style><h1>家居社交陪伴 VQA · 120题 / 110张图</h1><p>本轮224×224，W8混合精度NPU视觉塔，原生投影输出64视觉token；Intern语言推理同样在NPU。原图整体缩放，无裁脸。图片按实际像素大小展示。评分为单轮人工核心事实判断，非官方VQAv2分数；表情只指可见特征。本轮质量基准为视觉与语言分阶段常驻；并发实验另立报告。人物关系仅问可见共同活动，不推断亲属或身份。</p><label>筛选：<select onchange="document.querySelectorAll(&quot;section&quot;).forEach(s=>s.hidden=this.value!==&quot;all&quot;&&s.dataset.category!==this.value)"><option value="all">全部</option>'+''.join(f'<option value="{k}">{v}</option>' for k,v in labels.items())+'</select></label>'+''.join(sections)
(P/'GALLERY.html').write_text(page)
print(json.dumps(summary,ensure_ascii=False,indent=2))
