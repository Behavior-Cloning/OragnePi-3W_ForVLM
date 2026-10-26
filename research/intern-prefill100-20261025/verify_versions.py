from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent;repo=R.parent.parent
M=json.loads((R/'VERSION_MANIFEST.json').read_text())
assert M['schema_version']==3 and not M['goal_verified'] and not M['production_default_changed']
V={v['id']:v for v in M['versions']};assert len(V)==len(M['versions'])
checked=[]
for v in V.values():
 if 'report' not in v:continue
 p=(R/v['report']).resolve();assert p.is_file(),str(p)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==v['report_sha256'],v['id']+' report hash drift'
 c=v.get('source_commit') or v.get('report_commit');assert c,v['id']+' missing commit'
 z=subprocess.run(['git','show',c+':'+str(p.relative_to(repo))],cwd=repo,capture_output=True)
 assert z.returncode==0 and hashlib.sha256(z.stdout).hexdigest()==v['report_sha256'],v['id']+' committed report differs'
 checked.append(v['id'])
a=V['c448-original-tail8-r4'];C=json.loads((R/'c448-matrix-r4/config.json').read_text())
assert C['binary_sha256']==a['binary_sha256']
assert hashlib.sha256((R/'c448-matrix-r4/config.json').read_bytes()).hexdigest()==a['config_sha256']
assert hashlib.sha256((R/'c448-matrix-r4/inputs.json').read_bytes()).hexdigest()==a['inputs_sha256']
A=json.loads((R/a['analysis']).read_text())
assert len(A['performance'])==24 and len(A['diagnostics'])==6
for c in A['performance']:
 assert c['system_body']+c['system_template']==c['system_total']==c['reused']
 assert c['dynamic_body']+c['dynamic_template']==c['new_prefill']
 assert c['total_prompt']==c['reused']+c['new_prefill'] and c['output']==32
assert all(d['finite'] and d['same_bytes_as_original'] and d['bytes']==2426784 for d in A['diagnostics'])
assert len(A['closed'])==3 and all(x['metrics']['driver_bytes']==0 for x in A['closed'])
assert M['selections']['quality_qualified_best'] is None and not a['quality_qualified'] and not a['default']
for sel in ['performance_candidate','latest_original_quantization_checked']:
 assert M['selections'][sel]['id'] in V
v=V['outlier8-c448-full-r1'];d=R/'down-outlier-split-r2/full-test'
for name,field in [('ANALYSIS.json','analysis_sha256'),('config.json','config_sha256'),('inputs.json','inputs_sha256')]:
 assert hashlib.sha256((d/name).read_bytes()).hexdigest()==v[field]
B=json.loads((d/'ANALYSIS.json').read_text());assert len(B['performance'])==24 and len(B['diagnostics'])==6
for row in B['performance']:
 assert row['system_body']+5==row['system_total']==row['reused'] and row['dynamic_body']+8==row['new_prefill'] and row['output']==32
for diag in B['diagnostics']:
 f=d/'results'/diag['path'];assert f.stat().st_size==2426784 and hashlib.sha256(f.read_bytes()).hexdigest()==diag['sha256'] and diag['finite']
 if diag['variant']=='original-repeat':assert diag['same_bytes_as_original']
assert B['quality']['outlier8']['equal']==12 and B['fixed_feedback_equal']['outlier8']==23
assert len(B['closed'])==3 and all(x['metrics']['driver_bytes']==0 for x in B['closed'])
for rows,key in [(32,'main_nb_sha256'),(8,'tail_nb_sha256')]:
 f=R/'down-outlier-split-r2'/('full-c448-m'+str(rows))/'export_nbg_unify/network_binary.nb';assert hashlib.sha256(f.read_bytes()).hexdigest()==v[key]
assert v['quantization_changed'] and not v['quality_qualified'] and not v['default']
assert M['selections']['performance_candidate']['id']==v['id']
print(json.dumps({'status':'passed','reports_bound_to_commits':checked,'token_rows_checked':len(A['performance'])+len(B['performance']),'full_logit_records':len(A['diagnostics'])+len(B['diagnostics']),'matrices_checked':2,'default_changed':False},ensure_ascii=False))
