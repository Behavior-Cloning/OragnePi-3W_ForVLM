"""Deploy freshly compiled peak artifacts, verify both FP32 references, and measure five repeats.

Set OPI_BOARD_HOST, OPI_KNOWN_HOSTS and OPI_BOARD_HOST_LOCK for the deployment
host. Authentication is supplied by the caller; no password is stored here.
"""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
import argparse
import fcntl
parser = argparse.ArgumentParser(description='Verify a rebuilt peak stack on the board and repeat the archived performance protocol.')
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--historical', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--tag', required=True)
parser.add_argument('--case', required=True)
parser.add_argument('--reuse-tag', default='full-intern1b-c1280-cap1-headmse')
parser.add_argument('--build-report', type=Path, help='verified build-state.json; defaults to output/build-state.json')
args = parser.parse_args()
source, oldsource, output, model = (p.resolve() for p in (args.source, args.historical, args.output, args.model))
scripts = Path(__file__).resolve().parent
variant = 'w8-fused2-c1280-cap1-headmse'
prefill = 'w8-fused2-c1280-t16-cap1-productwide'
old = oldsource / variant / 'board-cap1-headmse-intern1b-h1024-n32'
case, tag = args.case, args.tag
board = source / variant / ('board-' + case)
e = os.environ.copy()
output.mkdir(parents=True, exist_ok=True)
# A second launcher must not overwrite the active run's state or logs.
_run_lock = (output / 'validation.lock').open('a')
try:
    fcntl.flock(_run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit('Validation already active for this output directory')
if (output / 'validation-state.json').exists():
    raise SystemExit('Existing validation evidence; use a new output directory and preserve this attempt')

state={'status':'waiting-for-build'}
def save(): (output/'validation-state.json').write_text(json.dumps(state,indent=2)+'\n')
def run(name,cmd):
 with (output/(name+'.log')).open('x') as log:
  state['status']=name;save();print('START',name,flush=True)
  subprocess.run([sys.executable,*map(str,cmd)],env=e,stdout=log,stderr=subprocess.STDOUT,check=True)
try:
 save();deadline=time.monotonic()+500
 while True:
  p=args.build_report or output/'build-state.json'
  if p.exists():
   b=json.loads(p.read_text())
   if b['status']=='compiled-awaiting-board-validation' and len(b['comparisons'])==28 and all(x.get('equal',x.get('files',{}).get('export_nbg_unify/network_binary.nb',{}).get('equal')) for x in b['comparisons']):break
  assert time.monotonic()<deadline
  time.sleep(5)
 run('fresh-deploy-full-logits',[scripts/'run_full_model.py','--source',source,'--dtype',variant,'--tag',tag,'--case',case,'--max-new','128','--decoder-group','12','--fusion-width','2','--prefill-variant',prefill,'--cached-prefix','1024','--fast-greedy','--group-head','--cpu-affinity','6,7','--prompt-folder',old,'--feedback-file',old/'feedback.i32','--reuse-tag',args.reuse_tag,'--deploy-layers',*range(0,24,2),*range(24,28),'--reuse-prefill','--deploy-prefill-layers',*range(0,24,2)])
 with (board/'logits.f32').open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 historical_performance=json.loads((oldsource.parent/'cap1-headmse-trial-20261013/performance.json').read_text())
 expected=next(v['expected_logits_sha256'] for v in historical_performance['config']['variants'] if v['label']=='candidate')
 state.update(fresh_logits_sha256=digest,historical_logits_sha256=expected,logits_equal=digest==expected);save()
 assert digest==expected,'fresh logits differ; diagnose before performance'
 for ref in ['exported','original']:
  run('fresh-reference-'+ref,[scripts/'validate_full.py','--source',source,'--model',model,'--dtype',variant,'--case',case,'--reference-weights',ref])
 for suffix in ['-w8','']:
  d=json.loads((board/('full-validation'+suffix+'.json')).read_text());h=json.loads((old/('full-validation'+suffix+'.json')).read_text())
  state['precision'+suffix]={k:d[k] for k in ['relative_l2_all_logits','min_step_cosine','argmax_matches_fp32','mean_kl_nats','fixed_trajectory_nll']}
  state['precision'+suffix]['historical_l2']=h['relative_l2_all_logits']
  for metric in ('relative_l2_all_logits','min_step_cosine','argmax_matches_fp32','mean_kl_nats','fixed_trajectory_nll'):
   assert d[metric]==h[metric], 'Historical precision mismatch: '+metric
  save()
 run('fresh-five-repeat-performance',[scripts/'benchmark_variants.py','--source',source,'--reference',board,'--variant','fresh',tag,case,'12','2','--variant-reference','fresh',board,'--prefill','fresh','16','--fast-greedy','fresh','--group-head','fresh','--fixed-feedback','--cpu-affinity','6,7','--max-start-temp-mc','68000','--temperature-band-mc','55000','64999','--repeats','5','--experiment',tag,'--output',output/'performance.json'])
 perf=json.loads((output/'performance.json').read_text());state.update(status='complete',medians=perf['medians'],performance_status=perf['status']);save();print(json.dumps(state),flush=True)
except BaseException as exc:
 state.update(status='failed',error=str(exc));save();raise
