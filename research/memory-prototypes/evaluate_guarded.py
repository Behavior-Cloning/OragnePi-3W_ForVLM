"""Untuned extra retrieval probes, intentionally includes ambiguous negative cases.
The gate was specified before these probes; retain failures instead of tuning to them.
"""
import json,tempfile,time
from pathlib import Path
from memory_lab import Memory
from benchmark_memory import FACTS,QUERIES
HELD_OUT=[
 ('positive','记住的本人姓名是什么','name'),
 ('positive','能告诉我居住在哪座城市吗','city'),
 ('positive','我的饮食禁忌有哪些','allergy'),
 ('positive','机器人名字是什么','project'),
 ('positive','银杏项目是在星期几开会','meeting'),
 ('positive','母亲过生日送的礼物是什么','gift'),
 ('positive','去年备用钥匙的存放位置','keys'),
 ('positive','我的猫几岁了','cat'),
 ('positive','复诊预约在几点','doctor'),
 ('positive','周末去听的爵士演出在哪里','concert'),
 ('unknown','我的驾照有效期是哪天',None),
 ('unknown','我的手机号码是多少',None),
 ('unknown','弟弟在哪家公司上班',None),
 ('unknown','这台机器人的电池电压是多少',None),
 ('unknown','我喜欢的电影叫什么名字',None),
 ('unknown','母亲的银行卡密码是多少',None),
 ('unknown','我的项目预算是多少',None),
 ('unknown','请告诉我朋友的姓名',None),
]
def evaluate(m,queries):
 rows=[]
 for cat,q,expected in queries:
  start=time.perf_counter();got=[r['external_id'] for r in m.retrieve('owner',q)]
  rows.append({'category':cat,'query':q,'expected':expected,'retrieved':got,'hit':expected in got if expected else None,'latency_ms':(time.perf_counter()-start)*1000})
 return {'positive_hit':sum(r['hit'] is True for r in rows),'positive_count':sum(r['expected'] is not None for r in rows),'unknown_false_returns':sum(bool(r['retrieved']) for r in rows if r['expected'] is None),'unknown_count':sum(r['expected'] is None for r in rows),'trials':rows}
if __name__=='__main__':
 with tempfile.TemporaryDirectory() as folder:
  m=Memory(Path(folder)/'guarded.db','profile_fts_guarded')
  m.remember_many('owner',[{'id':k,'content':v,'created':1700000000+i} for i,(k,v) in enumerate(FACTS)],confirmed=True)
  print(json.dumps({'strategy':m.strategy,'original_probes':evaluate(m,QUERIES),'extra_probes':evaluate(m,HELD_OUT),'caveat':'small synthetic set; no calibration claim; ambiguous related-topic negatives are deliberate'},ensure_ascii=False,indent=2))
  m.close()
