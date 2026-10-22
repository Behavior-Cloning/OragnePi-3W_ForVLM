import json,hashlib,re,sys
from pathlib import Path
from collections import Counter
P=Path(__file__).resolve().parent
A=json.loads((P/'atoms.json').read_text());C=json.loads((P/'capabilities.json').read_text())['capabilities'];H=json.loads((P/'hardware-profile.json').read_text());M=json.loads((P/'experiment-manifest.json').read_text())
checks=[]
def check(name,result,detail=None):checks.append({'check':name,'pass':bool(result),'detail':detail})
check('128_atoms_zero_skills',len(A)==128 and all(a['type']=='atom' and 'steps' not in a for a in A))
check('new_namespace_ids',[a['id'] for a in A]==[f'R{i:03d}' for i in range(1,129)] and all(a['namespace']=='home_companion_atoms128_v2' for a in A))
for field in ['id','name','code','description','retrieval_text']:
 d=[x for x,n in Counter(a[field] for a in A).items() if n>1];check('unique_'+field,not d,d)
check('all_mechanical_no_display',all(a['hardware_class']=='mechanical' for a in A))
allowed={'head','waist','arm:left','arm:right','base'}
check('only_confirmed_body_parts',all(set(a['commanded_channels'])<=allowed for a in A))
check('absence_recorded',set(H['user_confirmed_absent'])=={'eyes','display','finger_actuators'})
check('no_unconfirmed_wrist_shoulder_legs',set(H['not_assumed'])=={'independent_wrist','independent_shoulder_elevation','high_dof_leg_gaits'})
forbid=['眼','眉','嘴','脸','屏','图标','手指','手腕','指尖','拇','掌','拳','耸肩','跳跃','单脚']
violations=[a['id'] for a in A if any(x in a['name'] for x in forbid)]
check('no_absent_actuator_in_action_names',not violations,violations)
check('capability_refs_exist',all(c in C for a in A for c in a['capability_required']))
check('no_absent_actuator_capability',not any(any(t in c for t in ['display','eyes','finger','wrist','shoulder','leg']) for c in C))
check('all_unmapped_disabled',all(a['implementation_status']=='unmapped' and not a['enabled_on_board'] for a in A))
check('capabilities_not_claimed_verified',all(c['status']=='unverified' and c['enabled'] is False for c in C.values()))
check('global_motion_lock',all('motion_body' in a['resource_locks'] and set(a['commanded_channels'])<=set(a['resource_locks']) for a in A))
check('coupling_parts_known',all(set(a['potentially_affected_body_parts'])<=allowed for a in A))
check('base_affects_whole_robot',all(set(a['potentially_affected_body_parts'])==allowed for a in A if 'base' in a['commanded_channels']))
check('waist_coupling_disclosed',all({'head','arm:left','arm:right'}<=set(a['potentially_affected_body_parts']) for a in A if 'waist' in a['commanded_channels']))
check('single_contract_no_steps',all(a['controller_contract']['sequential_subactions'] is False and a['repeat_count']==1 for a in A))
check('state_vs_path_contract',all(a['controller_contract']['kind']==('zero_motion_hold' if a['kind']=='hold' else 'single_synchronized_target' if a['kind']=='pose' else 'single_continuous_trajectory') for a in A))
check('required_lifecycle',all(a.get(k) for a in A for k in ['terminal_state','preconditions','parameters','coordinate_frame','cancel_contract']))
check('simultaneous_base_not_assumed',all(any('同时执行' in x and '不得自动拆分' in x for x in a['preconditions']) for a in A if 'base_combined_path' in a['capability_required']))
check('orbit_center_radius_explicit',all({'center_xy_m','radius_m'}<=set(a['parameters']) for a in A if a['code'].startswith('base_orbit_')))
check('manifest_sha_match',M['candidate_sha256']==hashlib.sha256((P/'atoms.json').read_bytes()).hexdigest())
check('no_benchmark_claim',M['run_status']=='not_run' and not M['previous_results_applicable'] and not M['board_execution_performed'])
md=(P/'ATOMIC_ACTIONS_128.md').read_text();check('all_128_documented',re.findall(r'^\| (R\d{3}) \|',md,re.M)==[a['id'] for a in A])
check('circle_translation_differs_from_turn_arc',all('base_planar_path' in a['capability_required'] and 'base_combined_path' not in a['capability_required'] and '朝向始终保持' in a['description'] for a in A if a['code'].startswith('base_orbit_')))
check('zero_hold_has_no_displacement_params',all(a['kind']=='hold' and not {'distance_m','yaw_rad','amplitude_scale','speed_scale'} & set(a['parameters']) for a in A if a['code']=='base_hold'))
check('entry_conditions_explicit',all(a['entry_pose_requirement'] for a in A))
report={'status':'pass' if all(x['pass'] for x in checks) else 'fail','candidate_sha256':M['candidate_sha256'],'count':len(A),'groups':dict(Counter(a['group'] for a in A)),'checks':checks,'semantic_uniqueness':'See independent audit. String uniqueness is not semantic uniqueness.','board_execution_performed':False}
(P/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':report['status'],'checks':len(checks),'failures':[x for x in checks if not x['pass']]},ensure_ascii=False))
sys.exit(0 if report['status']=='pass' else 1)
