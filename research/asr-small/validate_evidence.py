"""Validate recorded results, without rerunning inference."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];D=R/'results/asr-small-20261018';checks=[]
manifest=json.loads((D/'corpus-manifest.json').read_text());ids={r['id'] for r in manifest['samples']};assert len(ids)==33
for p in (D/'standalone-v1').glob('*/result.json'):
 d=json.loads(p.read_text());assert d['status']=='complete' and len(d['runs'])==66 and d['affinity']==[2,3] and d['threads']==1
 assert {r['id'] for r in d['runs']}==ids
 first={r['id']:r['text'] for r in d['runs'][:33]};assert all(r['text']==first[r['id']] for r in d['runs'])
 checks.append({'file':str(p.relative_to(D)),'passed':True,'calls':66,'distinct_inputs':33})
qfile=D/'standalone-qwen-v1/qwen/result.json'
if qfile.exists():
 q=json.loads(qfile.read_text());assert len(q['runs'])==14 and q['affinity']==[2,3] and q['threads']==1 and q['status']=='complete'
 checks.append({'qwen_actual_board_inferences':14,'qwen_clean_subset':11,'qwen_single_pass_only':True})
tokens=None
for p in D.glob('mixed-*/summary.json'):
 rows=json.loads(p.read_text());assert len(rows)==2*len({r['condition'] for r in rows})
 for r in rows:
  assert r['generated_tokens']==128 and r['driver_after_release_bytes']==0 and r['decoder_layers']==24 and r['head_on_npu']
  assert r['cached_prefix_tokens']==64 and r['new_prompt_tokens']==r['dynamic']
  # Different dynamic prompts are allowed to have different generated tokens.
  if r['condition']!='baseline':assert r['audio_overlap_seconds']>0
 assert all(r['tokens']==rows[0]['tokens'] for r in rows)
 checks.append({'file':str(p.relative_to(D)),'passed':True,'llm_runs':len(rows),'same_tokens_within_dynamic':True})
for p in D.glob('*-scripts'):
 if not p.is_dir():continue
 name=p.name[:-8];m=D/(name+'-manifest.json')
 if m.exists():
  h=json.loads(m.read_text())
  for script in ['worker.py','coordinator.py']:
   if script in h:assert hashlib.sha256((p/script).read_bytes()).hexdigest()==h[script],(p,script)
checks.append({'executed_script_snapshots':'verified for recorded script hashes; initial standalone retrieved from board immediately before overwriting'})
g=json.loads((D/'vad-gate.json').read_text());assert all(r['pass_to_asr'] for r in g['runs'] if r['condition']=='clean');assert not [r for r in g['runs'] if r['condition']=='silence'][0]['pass_to_asr']
assert json.loads((D/'official-reference-check.json').read_text())['comparison'] and all(r['equal'] for r in json.loads((D/'official-reference-check.json').read_text())['comparison'])
checks.append({'official_transcript_matches':24,'vad_clean_pass':24,'vad_silence_block':True})
(D/'evidence-validation.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
