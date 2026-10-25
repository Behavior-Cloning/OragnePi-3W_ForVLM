#!/usr/bin/env python3
"""Read actual A733 NN profile counters; never assign untraced engine or wall time."""
from pathlib import Path
import collections,hashlib,importlib.util,json,re,statistics,struct
ROOT=Path(__file__).resolve().parent
TASK=ROOT.parents[1]
RESULTS=TASK/'results/single-block-profile-r1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
sdk=module('sdk_nb_data_item',ROOT/'sdk-source/nb_data_item.py')
prior=module('prior_profile_npd',ROOT/'prior_profile_npd.py')
segments=[sdk.ItemSegHEAD,sdk.ItemSeg0,sdk.ItemSeg1,sdk.ItemSeg2,sdk.ItemSeg3,sdk.ItemSeg4,sdk.ItemSeg5,sdk.ItemSeg15]
summary={'scope':'SDK NN hardware counters; NOT all-engine wall-time decomposition', 'sdk_record_adaptation':'Use unmodified SDK segment readers for actual 8 x 16-byte record. Do not invent missing SEL20..23.', 'cases':[], 'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'sdk-source').glob('*.py')}}
for d in sorted(RESULTS.glob('*-npd')):
 p=next(d.glob('*.npd'));raw=p.read_bytes();old=prior.read_npd(p)
 magic,version,target,nets,cores=struct.unpack_from('<4sIIII',raw)
 name,offset,size,count,stride,netcores=struct.unpack_from('<64sIIIII',raw,256)
 assert magic==b'VNPD' and version==2 and target==0x1000003b and nets==1 and cores==netcores==1 and stride==128
 assert size==count*stride and offset+size==len(raw)
 rows=[];by_node=collections.defaultdict(lambda:collections.Counter())
 for i in range(count):
  buf=raw[offset+i*stride:offset+(i+1)*stride];param={}
  for n,cls in enumerate(segments):param.update(cls(buf[n*16:n*16+16]).param)
  overflow={k:v for k,v in param.items() if k.endswith('_OVFL') and v}
  assert not overflow,(i,overflow)
  assert param['CoreID']==0
  assert (param['NodeId'],param['NN_TOTAL_BUSY_CYCLE'],param['NN_TOTAL_READ_VALID_BW_DDR'],param['NN_TOTAL_WRITE_VALID_BW_DDR'],param['NN_TOTAL_MAC_COUNT'])==tuple(old['rows'][i][k] for k in ['node','cycles','read_bytes','write_bytes','mac_count'])
  rows.append(param);n=by_node[param['NodeId']];n['records']+=1
  for k,v in param.items():
   if k not in ['NodeId','CoreID','NN_LAYER_ID','NN_INSTR_INFO'] and not k.endswith('_OVFL'):n[k]+=v
 total=sum(x['NN_TOTAL_BUSY_CYCLE'] for x in rows)
 nodes=[dict(node_id=k,engine='unknown_without_matching_trace',cycle_share_of_observed_NN_counters=v['NN_TOTAL_BUSY_CYCLE']/total,**v) for k,v in by_node.items()]
 nodes.sort(key=lambda x:-x['NN_TOTAL_BUSY_CYCLE'])
 out={'case':d.name,'npd':str(p.relative_to(TASK)),'npd_sha256':sha(p),'format':{'magic':'VNPD','version':version,'target':hex(target),'record_bytes':stride,'count':count},'sdk_vs_prior_parser_all_records_identical':True,'summed_NN_busy_cycles':total,'unique_observed_nodes':len(nodes),'engine_time_fraction':None,'reason':'No matching node_trace_db.json; NN busy counters omit non-NN targets and are not elapsed time.','nodes':nodes,'raw_records':rows}
 (ROOT/(d.name+'.json')).write_text(json.dumps(out,indent=2)+'\n')
 io=json.loads((RESULTS/d.name.replace('-npd','-io')/'result.json').read_text());nd=json.loads((d/'result.json').read_text())
 v=io['io_profile'];row={k:out[k] for k in ['case','npd_sha256','format','sdk_vs_prior_parser_all_records_identical','summed_NN_busy_cycles','unique_observed_nodes','engine_time_fraction']}
 row.update(p50_io_run_ms=1000*statistics.median(x['run_s'] for x in v),p50_io_sync_ms=1000*statistics.median(x['input_flush_s']+x['output_invalidate_s'] for x in v),p50_io_wall_ms=1000*statistics.median(x['wall_s'] for x in v),npd_api_ms=1000*nd['mean_api_seconds'],npd_only_diagnostic=True,top5_observed_nodes=nodes[:5])
 row['io_sync_fraction_of_wall']=row['p50_io_sync_ms']/row['p50_io_wall_ms']
 summary['cases'].append(row)
summary['IO_OUTPUT_AUDIT_sha256']=sha(RESULTS/'IO_OUTPUT_AUDIT.json')
summary['limitations']=['NPD records report NN counters only; cannot assign full latency to CPU/TP/SHADER.', 'Without exact-binary node trace, node IDs have no safe op/engine mapping.', 'NPD API timings have profiling overhead and are excluded from pure timing.', 'Busy counters can overlap; their sum is not end-to-end latency.']
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'cases':[{k:v for k,v in x.items() if k not in ['top5_observed_nodes','format']} for x in summary['cases']]},indent=2))
