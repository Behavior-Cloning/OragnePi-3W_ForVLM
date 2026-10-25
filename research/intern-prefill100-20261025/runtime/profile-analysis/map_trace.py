#!/usr/bin/env python3
from pathlib import Path
import collections,hashlib,json,re
ROOT=Path(__file__).resolve().parent;TASK=ROOT.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
reports=[]
for layer in [0,2]:
 case=f'c512-m16-l{layer}';trace_dir=TASK/'graphs/trace-export'/case
 old=TASK/'graphs/cases'/case/'export_nbg_unify/network_binary.nb';new=trace_dir/'export_nbg_unify/network_binary.nb'
 assert sha(old)==sha(new), 'Cannot map NPD from a different binary'
 tracepath=trace_dir/'export/node_trace_db.json';sourcepath=trace_dir/'export/vnn_decoder.c'
 trace=json.loads(tracepath.read_text());source=sourcepath.read_text()
 labels=collections.defaultdict(list)
 for lid,name,op,uid in re.findall(r'lid\s+- (\S+)\s+var\s+-[^\n]+\s+name\s+- (\S+)\s+operation\s+- (\S+).*?NEW_VXNODE\([^;]+, (\d+)\);',source,re.S):
  labels[int(uid)].append({'lid':lid,'name':name,'op':op})
 mapping=collections.defaultdict(set);targets=collections.defaultdict(set)
 for item in trace:
  for node in item['profile_node_ids']:
   mapping[node['node_id']].add(item['uid']['id'])
   if 'target' in node:targets[node['node_id']].add(node['target'])
 report=json.loads((ROOT/(case+'-npd.json')).read_text());actual={x['node_id']:x for x in report['nodes']}
 by_op=collections.defaultdict(lambda:collections.Counter());mapped=[]
 for n in report['nodes']:
  uids=sorted(mapping[n['node_id']]);ls=[x for u in uids for x in labels[u]];ops=sorted({x['op'] for x in ls});op='+'.join(ops) or 'unmapped'
  row=dict(n,uids=uids,ops=ops,candidate_lids=sorted({x['lid'] for x in ls}),trace_targets=sorted(targets[n['node_id']]))
  mapped.append(row)
  by_op[op]['observed_nodes']+=1;by_op[op]['NN_BUSY_CYCLES']+=n['NN_TOTAL_BUSY_CYCLE']
 missing=[]
 for node,uids in mapping.items():
  if node not in actual:missing.append({'node_id':node,'uids':sorted(uids),'candidate_lids':sorted({x['lid'] for u in uids for x in labels[u]}),'trace_targets':sorted(targets[node])})
 down_nodes=sorted(n for n,uids in mapping.items() if 116 in uids)
 down_observed=[n for n in down_nodes if n in actual]
 total=report['summed_NN_busy_cycles']
 for v in by_op.values():v['share_of_observed_NN_busy']=v['NN_BUSY_CYCLES']/total
 out={'case':case,'binary_sha256':sha(new),'nb_equal_original':True,'trace_sha256':sha(tracepath),'vnn_sha256':sha(sourcepath),'target_field_present_in_trace':any(bool(v) for v in targets.values()),'engine_time_fraction':None,'observed_NN_cycles':total,'down':{'uid':116,'candidate_lids':[x['lid'] for x in labels[116]],'profile_nodes':down_nodes,'observed_NN_profile_nodes':down_observed,'NN_cycles':sum(actual[n]['NN_TOTAL_BUSY_CYCLE'] for n in down_observed),'wall_time_s':None,'engine':'unknown: this trace omits target'},'by_op':dict(sorted(by_op.items(),key=lambda x:-x[1]['NN_BUSY_CYCLES'])),'nodes':mapped,'trace_nodes_without_NPD':missing,'limitations':['UID116 repeats across both fused layers; labels kept as candidates, no guessed b0/b1 assignment.','NN cycle shares are not elapsed-time shares. Missing NN record does not identify TP/SHADER/SW/CPU.','The exact-binary node trace format has no target field, despite newer SDK reader support.']}
 (ROOT/(case+'-mapped.json')).write_text(json.dumps(out,indent=2)+'\n');reports.append({k:v for k,v in out.items() if k not in ['nodes','trace_nodes_without_NPD']})
(ROOT/'mapping-summary.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports,indent=2))
