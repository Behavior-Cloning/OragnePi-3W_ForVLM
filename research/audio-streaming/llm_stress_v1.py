#!/usr/bin/env python3
"""Long sequential LLM stress test with system + recalled memory + current user turn."""
import json,statistics,time,urllib.request
from pathlib import Path
import sys
OUT=Path(sys.argv[1]); OUT.mkdir(parents=True,exist_ok=False)
SCENES=[('早餐提醒','明天早上七点半提醒我吃药。','用户习惯工作日七点半起床，药物提醒需要简短明确。'),('天气出门','今天下午适合带伞吗？','用户在上海，通常下午下班回家，偏好直接给出行动建议。'),('饮水','我今天好像喝水不够。','用户希望机器人温和提醒，每次只给一个小建议。'),('家电','客厅空调开到多少度比较合适？','用户家里有老人，偏好节能且舒适的建议。'),('日程','今晚八点提醒我给家人打电话。','用户重视家庭联系，提醒内容要包含时间和事项。'),('菜谱','冰箱里有鸡蛋和西红柿，晚饭怎么做？','用户偏好十五分钟内完成的简单菜谱。')]
total=60; rows=[]
def snap():
    d={'time':time.time(),'temp_mC':int(Path('/sys/class/thermal/thermal_zone0/temp').read_text()),'loadavg':Path('/proc/loadavg').read_text().strip()}
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                if 'llama-server' in (p/'cmdline').read_text(errors='ignore'):
                    for l in (p/'status').read_text().splitlines():
                        if l.startswith(('VmRSS:','VmHWM:')):d['llama_'+l.split(':')[0]]=int(l.split()[1])
                    break
            except OSError:pass
    return d
def call(system,memory,user):
    body=json.dumps({'model':'qwen25-15b-q4','messages':[{'role':'system','content':system},{'role':'user','content':'召回记忆：'+memory+'\n本轮用户：'+user}],'stream':True,'temperature':0,'max_tokens':48},ensure_ascii=False).encode()
    req=urllib.request.Request('http://127.0.0.1:8081/v1/chat/completions',data=body,headers={'Content-Type':'application/json'}); t=time.monotonic(); first=None; parts=[]
    with urllib.request.urlopen(req,timeout=180) as resp:
        for raw in resp:
            line=raw.decode('utf-8','replace').strip()
            if not line.startswith('data: '):continue
            if line[6:]=='[DONE]':break
            o=json.loads(line[6:]); p=o.get('choices',[{}])[0].get('delta',{}).get('content','')
            if p:
                if first is None:first=time.monotonic()-t
                parts.append(p)
    return first,time.monotonic()-t,''.join(parts)
def main():
    system='你是家庭陪伴机器人。回答简洁、可靠；不确定时明确说明。'
    for i in range(total):
        name,user,mem=SCENES[i%len(SCENES)]; err=None
        try:first,complete,reply=call(system,mem,user)
        except Exception as e:first=complete=None;reply='';err=repr(e)
        r={'turn':i+1,'scene':name,'first_s':first,'complete_s':complete,'reply_chars':len(reply),'error':err,'resources':snap(),'input_chars':len(system)+len(mem)+len(user)}; rows.append(r); print(json.dumps(r,ensure_ascii=False),flush=True); time.sleep(.15)
    valid=[r for r in rows if r['first_s'] is not None]; summary={'version':'llm-stress-v1','status':'complete' if len(valid)==len(rows) else 'errors','turns':rows,'config':{'total':total,'input':'system prompt + recalled memory + current user turn','model':'Qwen2.5-1.5B Q4 CPU llama-server','max_tokens':48},'aggregate':{'p50_first_s':statistics.median(r['first_s'] for r in valid),'p95_first_s':sorted(r['first_s'] for r in valid)[int(.95*len(valid))-1],'p50_complete_s':statistics.median(r['complete_s'] for r in valid),'temp_max_C':max(r['resources']['temp_mC'] for r in rows)/1000,'errors':sum(bool(r['error']) for r in rows),'memavailable_kib_min':None}}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(summary['aggregate'],ensure_ascii=False))
main()
