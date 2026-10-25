from pathlib import Path
import copy,hashlib,json,py_compile,sys,tempfile,os
from tokenizers import Tokenizer
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P))
import board_fullconv as bench
import controller_fullconv as controller
import contract
cfg,remote,sources=controller.preflight(P/'config.fullconv.json','host-preflight-only')
assert len(sources)==8 and cfg['repeats']==1
contract.validate_aba(cfg)
doc=json.loads((P/'inputs.json').read_text());tok=Tokenizer.from_file(doc['tokenizer_path']);enc=lambda s:tok.encode(s,add_special_tokens=False).ids
cases=bench.validate_fixture(doc,enc);assert len(cases)==8
bad=[]
for label,fn in [
 ('unset-candidate-sha',lambda c:c['candidate'].update(candidate_sha256='UNSET')),
 ('wrong-capacity',lambda c:c['candidate'].update(capacity=512)),
 ('variant-capacity256',lambda c:c['variants'][1].update(capacity=256)),
 ('feedback-length',lambda c:c['feedback_tokens'].pop()),
 ('decode-hash-altered',lambda c:c['variants'][1]['expected_graph_sha256'].update({remote+'/overlay/layer2.nb':'0'*64})),
 ('wrong-overlay-link',lambda c:c['overlay_contract']['symlinks'].update({'layer0.nb':remote+'/prefill-layer2-conv.nb'})),
 ('repeat-not-original',lambda c:c['variants'][2]['expected_graph_sha256'].update({contract.BASE+'/layer0.nb':'0'*64})),
 ('speed-profile-on',lambda c:c['variants'][1]['env'].update(OPI_PROBE_NPD='1')),
 ('different-threads',lambda c:c['variants'][1].update(affinity='6,7')),
 ('wrong-variant-order',lambda c:c['variants'].reverse()),
 ('extra-diagnostic',lambda c:c.update(extra_diagnostic_case_ids=['s64-d64-2'])),
 ('repeat-count-changed',lambda c:c.update(repeats=2))]:
 c=copy.deepcopy(cfg);fn(c)
 try:contract.validate_aba(c)
 except AssertionError:bad.append(label)
 else:raise AssertionError('bad config accepted: '+label)
for label,fn in [('wrong-new',lambda d:d['performance'][0].update(expected_new_tokens=64)),('truncated',lambda d:d['performance'][0]['tokens'].pop()),('missing-cell',lambda d:d['performance'].pop()),('wrong-body',lambda d:d['performance'][0].update(body_tokens=63))]:
 d=copy.deepcopy(doc);fn(d)
 try:bench.validate_fixture(d,enc)
 except AssertionError:bad.append(label)
 else:raise AssertionError('bad fixture accepted: '+label)
# Exercise actual symlink verifier on a temporary host filesystem, after separately checking config contract.
with tempfile.TemporaryDirectory(prefix='fullconv-overlay-') as temp:
 d=Path(temp);overlay=d/'overlay';overlay.mkdir();candidate=d/'prefill-layer2-conv.nb';candidate.write_bytes(b'candidate')
 links={}
 for name in sorted(contract.FILES):
  target=candidate if name=='prefill-layer2.nb' else d/name
  if target!=candidate:target.write_bytes(name.encode())
  (overlay/name).symlink_to(target);links[name]=str(target)
 local={'candidate':{'candidate_bytes':candidate.stat().st_size,'candidate_sha256':contract.sha(candidate)},'overlay_contract':{'overlay':str(overlay),'candidate_remote':str(candidate),'symlinks':links}}
 original=contract.validate_aba;contract.validate_aba=lambda _:True
 try:
  proof=contract.verify_overlay(local);assert len(proof['files'])==29
  (overlay/'layer0.nb').unlink();(overlay/'layer0.nb').symlink_to(candidate)
  try:contract.verify_overlay(local)
  except AssertionError:bad.append('physical-wrong-symlink')
  else:raise AssertionError('wrong physical link accepted')
 finally:contract.validate_aba=original
for f in P.glob('*.py'):py_compile.compile(str(f),doraise=True)
report={'status':'host_preflight_pass_not_board_run','variants':3,'per_variant':{'performance':24,'natural':12,'diagnostic_requests':8,'diagnostic_steps':32},'total':{'performance':72,'natural':36,'diagnostic_requests':24,'diagnostic_steps':96},'body_lengths':sorted({c['body_tokens'] for c in doc['performance']}),'static_lengths':sorted({c['static_tokens'] for c in doc['performance']}),'actual_new_lengths':sorted({c['expected_new_tokens'] for c in doc['performance']}),'max_prompt_tokens':max(len(c['tokens']) for c in doc['performance']),'max_KV_with32':max(len(c['tokens'])+31 for c in doc['performance']),'rejected_bad_inputs':bad,'overlay_files':29,'only_replaced':'prefill-layer2.nb','candidate_sha256':cfg['candidate']['candidate_sha256'],'source_sha256':{f.name:contract.sha(f) for f in P.glob('*.py')},'fixture_sha256':contract.sha(P/'inputs.json')}
(P/'host-preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
