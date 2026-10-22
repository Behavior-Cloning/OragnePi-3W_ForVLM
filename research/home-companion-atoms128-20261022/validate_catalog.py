import json,hashlib,re,sys
from pathlib import Path
from collections import Counter,defaultdict
P=Path(__file__).resolve().parent
a=json.loads((P/'atoms.json').read_text()); cap=json.loads((P/'capabilities.json').read_text())['capabilities']
manifest=json.loads((P/'experiment-manifest.json').read_text())
checks=[]
def check(name,ok,detail=None): checks.append({'check':name,'pass':bool(ok),'detail':detail})
check('exactly_128_atoms_no_skills',len(a)==128 and all(x['type']=='atom' and 'steps' not in x for x in a))
check('stable_namespace_and_contiguous_ids',[x['id'] for x in a]==[f'H{i:03d}' for i in range(1,129)] and all(x['namespace']=='home_companion_atoms128_v1' for x in a))
for key in ['id','code','name','description','canonical_effect','retrieval_text']:
    c=Counter(x[key] for x in a);dups=[v for v,n in c.items() if n>1]
    check('unique_'+key,not dups,dups)
check('all_scope_household',all(x['scope']=='home_desktop_companion' for x in a))
check('all_unmapped_and_disabled',all(x['implementation_status']=='unmapped' and x['enabled_on_board'] is False for x in a))
check('capability_references_exist',all(c in cap for x in a for c in x['capability_required']))
check('no_false_hardware_inventory',all(x['board_status']=='unknown' and x['mapped_presets']==[] for x in cap.values()))
check('resources_have_exclusion',all(set(x['resources'])<=set(x['exclusive_resources']) for x in a))
check('mechanical_common_exclusion',all('motion_body' in x['exclusive_resources'] for x in a if x['hardware_class']=='mechanical'))
check('display_never_claims_mechanical_hardware',all(all(c.startswith('display_') for c in x['capability_required']) for x in a if x['hardware_class']=='display'))
check('required_contract_fields',all(x.get(k) for x in a for k in ['preconditions','parameters','terminal_state','cancel_contract','atomic_contract','noise','energy']))
check('event_vs_state_lifetime',all(x['effect_lifetime']==('cycle_once' if x['kind']=='cycle' else 'caller_bounded_hold') for x in a))
check('cycles_not_duration_variants',all('hold_ms' not in x['parameters'] and 'repeat_count' in x['parameters'] for x in a if x['kind']=='cycle'))
forbidden=['查文档','查历史','超时','连接WiFi','查天气','查时间','识别人脸','声源定位','启动ASR','检测目标','打开摄像头','关机','重启','播放音乐']
check('no_management_sensing_or_multistage_names',not any(w in x['name'] for x in a for w in forbidden))
check('manifest_matches_candidate_hash',manifest['candidate_sha256']==hashlib.sha256((P/'atoms.json').read_bytes()).hexdigest())
check('manifest_does_not_reuse_old_measurements',manifest['run_status']=='not_run' and manifest['previous_results_applicable'] is False and manifest['candidate_count']==128 and manifest['skills_included'] is False)
md=(P/'ATOMIC_ACTIONS_128.md').read_text()
check('all_128_human_rows_present',[m for m in re.findall(r'^\| (H\d{3}) \|',md,re.M)]==[x['id'] for x in a])
clusters=defaultdict(list)
for x in a:
    key=re.sub(r'(left|right)','SIDE',x['code']);clusters[key].append(x['id'])
mirrors=[{'key':k,'ids':v,'reason':'不同侧执行器或相反方向；保留两个ID，不以速度幅度区分'} for k,v in clusters.items() if len(v)>1]
check('left_right_clusters_are_pairs',all(len(x['ids'])==2 for x in mirrors))
check('palms_have_shape_precondition',all('both_palms_ready_and_maintained' in x['geometry_preconditions'] and {'palm_surface:left','palm_surface:right'} <= set(x['capability_required']) for x in a if x['code'] in ['bilateral_palms_join','bilateral_cup','bilateral_cover_eyes','bilateral_cover_mouth','bilateral_air_clap','bilateral_stop']))
check('physical_coupling_disclosed',all(x['coupling_status'].startswith('kinematics_unverified') and x['potentially_affected_body_parts'] for x in a if x['hardware_class']=='mechanical'))
check('torso_linkage_disclosed',all({'head','arm:left','arm:right'} <= set(x['potentially_affected_body_parts']) for x in a if x['group']=='上身'))
check('wrist_forearm_coupling_disclosed',all('arm:'+side in x['potentially_affected_body_parts'] for x in a for side in ['left','right'] if 'wrist:'+side in x['commanded_channels']))
report={'status':'pass' if all(x['pass'] for x in checks) else 'fail','candidate_sha256':manifest['candidate_sha256'],
 'count':len(a),'by_category':dict(Counter(x['group'] for x in a)), 'by_hardware':dict(Counter(x['hardware_class'] for x in a)),
 'checks':checks,'left_right_clusters':mirrors,'semantic_review_scope':'Exact-field checks do not prove semantic uniqueness; see independent-audit.json for full semantic review.',
 'board_execution_performed':False,'model_benchmark_performed':False}
(P/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':report['status'],'checks':len(checks),'failed':[x for x in checks if not x['pass']],'left_right_pairs':len(mirrors)},ensure_ascii=False))
sys.exit(0 if report['status']=='pass' else 1)
