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
assert not v['quality_qualified']  # retained historical performance candidate
# A newer experiment is explicitly not a best-version promotion.
c=V['compact-fullquery-r3'];d=R/'compact-fullquery-r3/full-test';C=json.loads((d/'ANALYSIS.json').read_text())
for name,field in [('ANALYSIS.json','analysis_sha256'),('config.json','config_sha256'),('inputs.json','inputs_sha256')]:
 assert hashlib.sha256((d/name).read_bytes()).hexdigest()==c[field]
assert len(C['performance'])==12 and len(C['diagnostics'])==12 and C['old_runtime_control']['same_bytes']
for row in C['performance']:
 assert row['system_body']+5==row['system_total']==row['reused'] and row['dynamic_body']+8==row['new_prefill'] and row['output']==32
for diag in C['diagnostics']:
 f=d/'results'/diag['path'];assert f.stat().st_size==2426784 and hashlib.sha256(f.read_bytes()).hexdigest()==diag['sha256'] and diag['finite']
 if diag['variant']=='parent-repeat':assert diag['same_bytes_as_original']
assert C['quality']['compact']['equal']==11 and C['fixed_feedback_equal']['compact']==11
assert len(C['closed'])==3 and all(x['metrics']['driver_bytes']==0 for x in C['closed'])
assert c['kind']=='evaluated-no-clear-gain-not-promoted' and not c['default'] and not c['quality_qualified']
assert any(x['candidate']==c['id'] and x['decision']=='not-promoted' for x in M['promotion_history'])
# M64 full-model candidate: performance tracking never certifies semantic correctness.
v=V['compact-m64-r4'];d=R/'compact-m64-r4/full-test'
for name,key in [('REPORT.md','report_sha256'),('ANALYSIS.json','analysis_sha256'),('QUALITY_REVIEW.json','quality_review_sha256'),('config.json','config_sha256'),('inputs.json','inputs_sha256'),('deployment-sha.json','deployment_sha256')]:
 assert hashlib.sha256((d/name).read_bytes()).hexdigest()==v[key],name+' M64 drift'
D=json.loads((d/'ANALYSIS.json').read_text());Q=json.loads((d/'QUALITY_REVIEW.json').read_text());I=json.loads((d/'inputs.json').read_text())
assert D['controller']['status']=='complete' and D['controller']['returncode']==0
assert len(D['performance'])==39 and len(D['diagnostics'])==18 and D['old_runtime_control']['same_bytes']
assert D['counts']['measure']==117 and D['counts']['quality']==36
expected={(32,b) for b in [32,64,128,256]}|{(64,b) for b in [32,64,128,192,256]}|{(128,b) for b in [64,128,192,256]}
for variant in ['parent','compact','parent-repeat']:
 rows=[x for x in D['performance'] if x['variant']==variant]
 assert {(x['system_total'],x['dynamic_body']) for x in rows}==expected
 for x in rows:
  assert x['system_body']+x['system_template']==x['system_total']==x['reused']
  assert x['dynamic_body']+x['dynamic_template']==x['new_prefill'] and x['dynamic_template']==8
  assert x['total_prompt']==x['new_prefill']+x['reused'] and x['output']==32 and x['ttft_seconds']['n']==3
 for c in D['quality'][variant]['cases']:
  assert c['dynamic_body']+8==c['actual_new'] and 2<=c['output_tokens']<=5
 assert Q['counts'][variant]==dict(fulfills_requested_action=6,omits_requested_action=5,conflicts_with_explicit_prohibition=1)
assert len(Q['rows'])==36 and not Q['quality_qualified']
assert Q['inputs_sha256']==v['inputs_sha256'] and Q['analysis_sha256']==v['analysis_sha256']
for diag in D['diagnostics']:
 f=d/'results'/diag['path'];assert f.stat().st_size==2426784 and hashlib.sha256(f.read_bytes()).hexdigest()==diag['sha256'] and diag['finite']
 if diag['variant']=='parent-repeat':assert diag['same_bytes_as_original']
assert D['quality']['compact']['equal']==10 and D['fixed_feedback_equal']['compact']==37
assert len(D['closed'])==3 and all(x['metrics']['driver_bytes']==0 for x in D['closed'])
assert M['selections']['performance_candidate']['id']==v['id'] and not v['quality_qualified'] and not v['default']
assert M['selections']['latest_evaluated_candidate']['id']==v['id'] and not M['selections']['latest_evaluated_candidate']['promoted']
assert v['quantization_changed_relative_to_parent'] is False and v['parent_quantization_changed_from_original'] is True
assert {(x['static_tokens'],x['body_tokens']) for x in I['performance']}==expected and len(I['performance'])==39
for x in D['performance']:
 if x['variant']=='compact' and x['dynamic_body']>=64:
  controls=[r['ttft_seconds']['p50'] for r in D['performance'] if r['variant']!='compact' and (r['system_total'],r['dynamic_body'])==(x['system_total'],x['dynamic_body'])]
  assert x['ttft_seconds']['p50']<min(controls)
# Local screen remains separate from full-model ranking.
v=V['softmax-layout-r1'];d=R/'softmax-layout-r1';E=json.loads((d/'ANALYSIS.json').read_text())
assert hashlib.sha256((d/'ANALYSIS.json').read_bytes()).hexdigest()==v['analysis_sha256']
assert len(E['results'])==3 and all(r['n']==30 and r['finite'] and r['same_bytes'] and r['driver0'] for r in E['results'])
for r in E['results']:
 f=d/'results'/(r['variant']+'.f32');assert f.stat().st_size==360448 and hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256']
assert E['results'][1]['p50_ms']>max(E['results'][0]['p50_ms'],E['results'][2]['p50_ms'])
assert not v['quality_qualified'] and not v['default']

for v in V.values():
 if 'git_tag' in v:
  tag=subprocess.check_output(['git','rev-parse',v['git_tag']+'^{commit}'],cwd=repo,text=True).strip();assert tag==v['source_commit'],v['id']+' tag moved or mismatch'
print(json.dumps({'status':'passed','reports_bound_to_commits':checked,'token_rows_checked':len(A['performance'])+len(B['performance'])+len(C['performance'])+len(D['performance']),'full_logit_records':len(A['diagnostics'])+len(B['diagnostics'])+len(C['diagnostics'])+len(D['diagnostics']),'matrices_checked':4,'default_changed':False,'latest_promoted':False},ensure_ascii=False))
