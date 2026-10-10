#!/usr/bin/env python3
"""Offline-only validation and preparation of companion conversations and blank review sheets."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIMENSIONS = ['naturalness', 'persona_consistency', 'local_coherence', 'adaptation']


def load_suite():
    path = ROOT/'tests/companion/phase1.json'
    suite = json.loads(path.read_text())
    persona = (path.parent/suite['persona_file']).read_text().strip()
    scenarios = suite['scenarios']
    ids = [s['id'] for s in scenarios]
    assert len(ids) == len(set(ids)), 'Duplicate scenario IDs'
    assert set(suite['smoke_ids']) <= set(ids), 'Unknown smoke scenario'
    assert persona and suite['defaults']['thinking'] is False
    for scenario in scenarios:
        assert scenario['split'] in ('development','holdout')
        assert scenario['title'] and scenario['failure_signals']
        assert len(scenario['turns']) == 4
        for turn in scenario['turns']:
            assert isinstance(turn['user'],str) and turn['user'].strip()
            assert isinstance(turn['expect'],str) and turn['expect'].strip()
    assert all(s['split']=='development' for s in scenarios if s['id'] in suite['smoke_ids'])
    return suite, persona


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=['smoke','development','holdout','all'],default='smoke')
    parser.add_argument('--seeds',nargs='+',type=int,default=[42])
    parser.add_argument('--exclude', nargs='*', default=[], help='Scenario IDs already run with identical settings')
    parser.add_argument('--output-dir',type=Path,help='Write a new plan and blank score sheet; no network or inference')
    args=parser.parse_args()
    suite,persona=load_suite()
    selected=[s for s in suite['scenarios'] if args.split=='all' or s['split']==args.split or
              (args.split=='smoke' and s['id'] in suite['smoke_ids'])]
    assert set(args.exclude) <= {s['id'] for s in suite['scenarios']}, 'Unknown excluded scenario'
    selected = [s for s in selected if s['id'] not in args.exclude]
    tasks=[];review=[]
    for seed in args.seeds:
        for scenario in selected:
            task_id=f"{scenario['id']}-seed{seed}"
            # Only persona and user turns go in the execution plan. Reviewer expectations never go to the model.
            tasks.append({'id':task_id,'scenario_id':scenario['id'],'seed':seed,
                          'system_prompt':persona,'user_turns':[t['user'] for t in scenario['turns']],
                          'protocol':'Reset history per scenario; append each actual assistant reply before the next user turn.'})
            review.append({'id':task_id,'scenario_id':scenario['id'],'title':scenario['title'],
                           'anonymous_model':None,'reviewer':None,'valid_run':None,
                           'scores':{d:None for d in DIMENSIONS},'critical_events':[],
                           'turns':[{'turn':i+1,'user':t['user'],'expect':t['expect'],
                                     'actual_answer':None,'evidence':None} for i,t in enumerate(scenario['turns'])],
                           'failure_signals':scenario['failure_signals'],
                           'would_continue_chatting':None,'notes':None})
    digest=hashlib.sha256((ROOT/'tests/companion/phase1.json').read_bytes()+persona.encode()).hexdigest()
    plan={'suite_version':suite['version'],'suite_sha256':digest,'split':args.split,'excluded_ids':args.exclude,
          'settings':{**suite['defaults'],'seeds':args.seeds},'tasks':tasks,
          'execution_status':'NOT_RUN','turn_count':sum(len(t['user_turns']) for t in tasks)}
    if args.output_dir:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        targets=[args.output_dir/'plan.json',args.output_dir/'review-blank.json']
        if any(p.exists() for p in targets):raise SystemExit('Choose a fresh directory; existing plans/scores will not be overwritten.')
        for path,content in zip(targets,[plan,{'suite_sha256':digest,'status':'UNSCORED','items':review}]):
            with path.open('x') as output: json.dump(content,output,ensure_ascii=False,indent=2)
    print(json.dumps({'validated_scenarios':len(suite['scenarios']),
                      'total_user_turns':sum(len(s['turns']) for s in suite['scenarios']),
                      'selected_tasks':len(tasks),'selected_turns':plan['turn_count'],
                      'suite_sha256':digest,'status':'PREPARED_ONLY_NOT_RUN'},ensure_ascii=False))


if __name__=='__main__':main()
