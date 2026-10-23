"""Same Chinese calibration protocol, Q4 source only; no board access."""
from pathlib import Path
import os,subprocess,json,hashlib,shutil,time
R=Path(__file__).resolve().parent;P='/opt/opi-lab/export-venv/bin/python';B=Path('/opt/opi-lab/llama32-q8-quant-probe-20261023/q4-control');E=R/'q4-control';E.mkdir(exist_ok=True);B.mkdir(exist_ok=True)
env=dict(os.environ,LLAMA_GGUF_MODEL=str(R.parent/'vlm-molmo-paligemma-20261023/paligemma/llama-zh/FuseChat-Llama-3.2-1B-Instruct-Q4_K_M.gguf'),LLAMA_BUILD_ROOT=str(B),PROBE_RECORD_ROOT=str(E))
def run(label,cmd):
 assert shutil.disk_usage('/mnt/f').free>9*1024**3
 with (E/(label+'.log')).open('w') as f:z=subprocess.run([P]+cmd,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=1000)
 if z.returncode:raise RuntimeError(label+' failed')
 print(label,'complete',flush=True)
run('prepare',[str(R/'prepare.py'),'--layer0-only'])
a=json.loads((E/'fixtures-manifest.json').read_text());b=json.loads((R/'fixtures-manifest.json').read_text());assert a['source_tokens']==b['source_tokens'];assert [(x['case_id'],x['position']) for x in a['decoder_fixtures']]==[(x['case_id'],x['position']) for x in b['decoder_fixtures']]
S=B/'scratch';C=S/'compiled';run('export',[str(R/'export_one.py'),'--index','0','--stage',str(S)])
run('calibrate',[str(R/'import_calibrate.py'),'--source',str(S),'--output',str(C),'--dtype','int16','--timeout','300'])
run('compile',[str(R/'compile_w8.py'),'--base',str(C),'--output',str(S/'allw8'),'--float-sensitive','--float-residual','--timeout','300'])
F=B/'nbg';F.mkdir(exist_ok=True);src=S/'allw8/export_nbg_unify/network_binary.nb';dest=F/'layer0-allw8.nb';shutil.copyfile(src,dest)
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
assert sha(src)==sha(dest)
for fn in ['decoder.json','decoder.quantize','conversion.json','export.log']:shutil.copyfile(S/'allw8'/fn,E/fn)
shutil.copyfile(S/'export-manifest.json',E/'export-manifest.json');manifest=dict(path=str(dest),bytes=dest.stat().st_size,sha256=sha(dest),source=a['source'],source_sha256=a['source_sha256'],paired_q8_source=b['source_sha256'],same_token_positions_verified=True,source_only_control='Same Chinese text, positions, precision flags, capacity and data-generation code; source Q4 vs Q8 differs');(E/'manifest.json').write_text(json.dumps(manifest,indent=2));shutil.rmtree(S);print(json.dumps(manifest),flush=True)
