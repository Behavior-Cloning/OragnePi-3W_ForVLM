"""Synthetic retrieval evidence and resource cost; not a user-answer accuracy score."""
import argparse,json,os,random,resource,sqlite3,statistics,subprocess,sys,time
from pathlib import Path
from memory_lab import Memory,STRATEGIES

FACTS=[
 ('name','我叫林然'),('city-old','我住在南京'),('city','我现在住在苏州'),
 ('drink','我最喜欢喝乌龙茶'),('allergy','我对花生过敏'),('project','我的项目叫星芽'),
 ('book','上周读完了三体，最喜欢黑暗森林这一部'),
 ('concert','周六和小夏去听了爵士音乐会，场地在滨江剧院'),
 ('meeting','银杏项目会议改到周四下午三点，地点是二楼会议室'),
 ('keys','备用钥匙放在玄关左侧抽屉里'),
 ('cat','家里的猫叫芝麻，今年四岁'),
 ('gift','母亲生日准备送一本植物图鉴'),
 ('travel','十一想去云南大理看洱海，计划坐高铁'),
 ('sleep','夜里喝咖啡会睡不着，下午一般不碰咖啡'),
 ('repair','机器人左侧轮子昨天有异响，换了螺丝后恢复正常'),
 ('food','周日做了番茄牛腩，家人说味道偏咸'),
 ('doctor','牙科复诊安排在下周二上午九点'),
 ('exercise','喜欢晚上沿着运河慢跑半小时'),
 ('camera','书房摄像头暂时关闭，周末再调试'),
 ('secret','禁止把保险柜密码写进聊天记录'),
]
QUERIES=[
 ('identity','你应该怎么称呼我？','name'),('identity','我的名字是什么？','name'),
 ('update','我现在住在哪个城市？','city'),('update','记得我的居住地吗？','city'),
 ('preference','给我推荐喝什么饮品？','drink'),('safety','我有什么食物过敏？','allergy'),
 ('project','我的项目叫什么？','project'),
 ('lexical','三体里我最喜欢哪一部？','book'),
 ('lexical','爵士音乐会是在哪里听的？','concert'),
 ('lexical','银杏项目会议几点开始？','meeting'),
 ('lexical','备用钥匙放在哪儿？','keys'),
 ('lexical','家里的猫叫什么？','cat'),
 ('lexical','母亲生日准备送什么？','gift'),
 ('lexical','十一去哪里旅行？','travel'),
 ('lexical','为什么下午不喝咖啡？','sleep'),
 ('lexical','机器人左侧轮子怎么修好的？','repair'),
 ('lexical','番茄牛腩味道怎么样？','food'),
 ('lexical','牙科复诊是什么时间？','doctor'),
 ('lexical','我喜欢在哪儿慢跑？','exercise'),
 ('lexical','书房摄像头什么时候调试？','camera'),
 ('paraphrase','那场演出我和谁一起去的？','concert'),
 ('paraphrase','家里应急开门的东西藏哪了？','keys'),
 ('paraphrase','我的宠物今年多大？','cat'),
 ('paraphrase','晚上失眠可能和什么饮品有关？','sleep'),
 ('unknown','我的护照号码是多少？',None),
 ('unknown','我毕业于哪所大学？',None),
 ('unknown','我的银行卡余额是多少？',None),
 ('unknown','我们第一次去南极是哪一年？',None),
]

def percentile(xs,q):return sorted(xs)[min(len(xs)-1,int((len(xs)-1)*q))]
def rss_kib():
 return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

def io():
 try:return {k:int(v) for k,v in (s.split(':') for s in Path('/proc/self/io').read_text().splitlines())}
 except OSError:return {}

def worker(strategy,count,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=False)
 start=time.perf_counter();m=Memory(out/'memory.db',strategy,max_items=count+100)
 opened_ms=(time.perf_counter()-start)*1000
 rows=[{'id':i,'content':text,'created':1700000000+j} for j,(i,text) in enumerate(FACTS)]
 colors=['蓝色','黄色','紫色','白色'];objects=['电池','纸箱','电线','木板','支架','胶带'];locations=['仓库','车库','阳台','门厅']
 for i in range(count-len(rows)):
  text=f'库存记录{i:05d}：{colors[i%4]}{objects[i%6]}已放到{locations[i%4]}，编号{i+10000}，数量{i%9+1}。'
  rows.append({'id':f'noise-{i}','content':text,'created':1700000100+i})
 before=io();cpu=time.process_time();t=time.perf_counter()
 m.remember_many('owner',rows,confirmed=True)
 bulk={'elapsed_ms':(time.perf_counter()-t)*1000,'cpu_ms':(time.process_time()-cpu)*1000,'rows':count}
 # Other user's exact-looking data must not be returned.
 m.remember_many('other',[{'id':'other','content':'我叫陈星','created':1900000000}],confirmed=True)
 writes=[]
 for i in range(12):
  t=time.perf_counter();m.remember_many('owner',[{'id':f'commit-{i}','content':f'完成独立写入事务{i}','created':1800000000+i}],confirmed=True);writes.append((time.perf_counter()-t)*1000)
 m.close()
 t=time.perf_counter();m=Memory(out/'memory.db',strategy,max_items=count+100);reopen_ms=(time.perf_counter()-t)*1000
 t=time.perf_counter();m.retrieve('owner',QUERIES[0][1]);first_query_ms=(time.perf_counter()-t)*1000
 trials=[];latencies=[];cpus=[]
 order=list(range(len(QUERIES)));random.Random(20261018).shuffle(order)
 for repeat in range(3):
  for index in order:
   category,q,expected=QUERIES[index];ct=time.process_time();t=time.perf_counter()
   result=m.retrieve('owner',q);latencies.append((time.perf_counter()-t)*1000);cpus.append((time.process_time()-ct)*1000)
   assert all(r['namespace']=='owner' for r in result)
   if repeat==0:trials.append({'category':category,'query':q,'expected':expected,'retrieved':[r['external_id'] for r in result],'hit':expected in [r['external_id'] for r in result] if expected else None})
 quality={}
 for cat in sorted({q[0] for q in QUERIES}):
  tests=[x for x in trials if x['category']==cat]
  quality[cat]={'count':len(tests),'hits':sum(x['hit'] is True for x in tests),'empty':sum(not x['retrieved'] for x in tests)}
 positive=[x for x in trials if x['expected']]
 result={'strategy':strategy,'records':count,'sqlite':sqlite3.sqlite_version,'open_ms':opened_ms,'reopen_ms':reopen_ms,'first_query_after_reopen_ms':first_query_ms,'first_query_scope':'same process, DB connection reopened; OS page cache not flushed',
  'bulk_insert':bulk,'commit_ms_p50':statistics.median(writes),'commit_ms_p95':percentile(writes,.95),'retrieval_ms_p50':statistics.median(latencies),'retrieval_ms_p95':percentile(latencies,.95),'query_cpu_ms_p50':statistics.median(cpus),'query_cpu_ms_p95':percentile(cpus,.95),'peak_process_rss_kib':rss_kib(),'quality_by_category':quality,'hit_at_3':sum(x['hit'] for x in positive)/len(positive),'positive_queries':len(positive),'unknown_false_returns':sum(bool(x['retrieved']) for x in trials if x['expected'] is None),'unknown_queries':4,'trials':trials}
 assert m.db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
 m.close();after=io();result['process_io_delta']={k:after.get(k,0)-before.get(k,0) for k in ['read_bytes','write_bytes']}
 result['db_bytes']=(out/'memory.db').stat().st_size
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='trials'},ensure_ascii=False),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--strategy',choices=STRATEGIES);p.add_argument('--count',type=int);a=p.parse_args()
 if a.strategy:worker(a.strategy,a.count,a.out)
 else:
  out=Path(a.out);out.mkdir(parents=True,exist_ok=False);results=[]
  for n in (500,5000):
   for strategy in STRATEGIES:
    path=out/f'{strategy}-{n}'
    subprocess.run([sys.executable,__file__,'--out',str(path),'--strategy',strategy,'--count',str(n)],check=True)
    results.append(json.loads((path/'result.json').read_text()))
    (out/'summary.json').write_text(json.dumps({'status':'running','results':results},ensure_ascii=False,indent=2)+'\n')
  (out/'summary.json').write_text(json.dumps({'status':'complete','results':results},ensure_ascii=False,indent=2)+'\n')
