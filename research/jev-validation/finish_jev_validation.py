"""Continue this experiment queue and record measured outcomes, never approve model quality."""
from pathlib import Path
import os,time,subprocess,json,datetime,traceback
r=Path(__file__).resolve().parent
status=r/'results/continuation-status.json'
def record(phase,**kw):
 d=dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=phase,**kw);status.write_text(json.dumps(d,ensure_ascii=False,indent=2));print(d,flush=True)
record('waiting_for_optimizer',pid=5972)
while Path('/proc/5972').exists():time.sleep(15)
try:
 record('resuming_missing_attempts')
 # Keep completed/partial attempts intact; recover only configurations never started.
 z=subprocess.run(['python3','-u',str(r/'run_board_optimization.py')],env={**os.environ,'JEV_RESUME':'1'},stdout=open(r/'results/resume-optimization.log','w'),stderr=subprocess.STDOUT)
 record('resume_returned',controller_exit=z.returncode)
 record('selecting_performance_configs')
 subprocess.run(['python3',str(r/'compare_optimization.py')],stdout=open(r/'results/compare-optimization.log','w'),stderr=subprocess.STDOUT,check=True)
 runs=json.loads((r/'results/optimization-comparison.json').read_text());configs={};why={}
 for model in ['nano33','laya','jev08']:
  eligible=[x for x in runs if x['model']==model and x['complete'] and x['n']>=20 and x['quality_n']==8]
  baselines=[x for x in eligible if x['label']=='a76-t2']
  if not baselines:why[model]={'status':'missing_complete_baseline'};continue
  base=baselines[-1]
  if model=='nano33' and base['native_n']!=18:why[model]={'status':'missing_native_controls'};continue
  parity=[x for x in eligible if x['answers']==base['answers'] and x['timing_answers']==base['timing_answers'] and (model!='nano33' or (x['native_n']==base['native_n']==18 and x['native_answers_sequence']==base['native_answers_sequence']))]
  chosen=min(parity or [base],key=lambda x:x['p50']);env={k:v for k,v in chosen['loaded']['environment'].items() if v is not None}
  configs[model]={'threads':chosen['loaded']['optimization']['threads'],'cores':chosen['loaded']['affinity'],'env':env}
  why[model]={'source_file':chosen['file'],'label':chosen['label'],'seconds_p50':chosen['p50'],'quality_correct':chosen['quality_correct'],'quality_n':8,'note':'Chosen by 10-candidate P50 with 20 timed answers and eight quality answers matching baseline; Nano also requires all 18 native control answers. Does not optimize direct100, approve deployment, or prove safety.'}
 (r/'results/selected-configs.json').write_text(json.dumps(configs,indent=2));(r/'results/selection-rationale.json').write_text(json.dumps(why,ensure_ascii=False,indent=2))
 record('running_isolated_sustained_pairs',models=list(configs),selection=why)
 z=subprocess.run(['python3','-u',str(r/'run_final_pairs.py')],stdout=open(r/'results/run-final-pairs.log','w'),stderr=subprocess.STDOUT)
 record('collecting',final_pairs_controller_exit=z.returncode)
 subprocess.run(['python3',str(r/'summarize_results.py')],stdout=open(r/'results/summarize-final.log','w'),stderr=subprocess.STDOUT,check=True)
 summary=json.loads((r/'results/summary.json').read_text());out=['## 自动回收的优化初筛结果（非最终部署验收）','', '|模型|配置|完成|计时n|P50秒|P95秒|质量正确/8|相对基线速度倍数|', '|---|---|---|---:|---:|---:|---:|---:|']
 for x in runs:out.append('|{}|{}|{}|{}|{}|{}|{}/{}|{}|'.format(x['model'],x['label'],x['complete'],x['n'],round(x['p50'],4) if x['p50'] else '未测',round(x['p95'],4) if x['p95'] else '未测',x['quality_correct'],x['quality_n'],round(x.get('speedup',0),3)))
 out+=['','每格仅20次计时与8条质量初筛；原始JSONL、资源峰值、决策差异见optimization-comparison.json。Nano中文输入失效结论不因速度改善而改变。','', '### 优胜配置实际共驻回收', '']
 selected={k:v for k,v in summary.items() if k.startswith('co-resident-optimized-')}
 for name,rec in selected.items():
  out.append('- '+name+'：文件结束标记='+str(rec['completed'])+'；错误数='+str(len(rec['errors']))+'；有效条件='+str({k:v['n'] for k,v in rec.get('pairs',{}).items()}))
 out+=['','持续决策窗口的具体时间、每次case和LLM指标保留在原始记录中；需人工审计后再决定推荐。未涵盖视觉encoder、全语音链路、多模态OneJev、长时稳定性与1000次尾延迟，不将这些TODO勾选。']
 p=r/'JEV_DEPLOYMENT_20261021.md';s=p.read_text();marker='## 自动回收的优化初筛结果（非最终部署验收）';s=s.split(marker)[0].rstrip();p.write_text(s+'\n\n'+'\n'.join(out)+'\n')
 hp=r.parents[1]/'docs/HANDOFF.md';hp.write_text('## Jev 后续队列已回收（'+datetime.datetime.now().isoformat()+'）\n\n优化及共驻自动回收见 research/jev-validation/JEV_DEPLOYMENT_20261021.md 最后一节与 results/continuation-status.json。仍需审计结果、补完整质量/长稳/视觉并同步 Mac 报告；远程下载独立运行，尚未由此脚本确认完成。\n\n'+hp.read_text())
 record('screening_pipeline_returned',final_pairs_controller_exit=z.returncode,returned_pair_files=list(selected),selection=why,full_project_complete=False)
except BaseException as e:
 record('continuation_error',error=repr(e),trace=traceback.format_exc());raise
