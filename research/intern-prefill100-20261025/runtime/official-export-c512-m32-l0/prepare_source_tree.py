from pathlib import Path
import shutil,json,hashlib
P=Path(__file__).resolve().parent;T=P.parent.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
proof=json.loads((P/'source-proof.json').read_text());plain=T/'graphs/cases/c512-m32-l0/export';affine=T/'graphs/candidates-w8/conv1x1-affine-c512-m32-l0/export';patched=T/'runtime/bitcast-c512-m32-l0/export'
for v,src in [('plain',plain),('bitcast',patched)]:
 E=P/v
 if E.exists():raise SystemExit('Refuse overwrite existing evidence: '+str(E))
 shutil.copytree(src,E,ignore=shutil.ignore_patterns('*.o','gen_nbg','*.nb','ports-*.json'))
 shutil.copy2(plain/'main.c',E/'main.c');shutil.copy2(affine/'makefile.linux',E/'makefile.linux')
 assert sha(E/'vnn_decoder.c')==proof['variants'][v]['c_sha256']
 assert sha(E/'decoder.export.data')==proof['variants'][v]['data_sha256']
 assert sha(E/'main.c')==proof['main_sha256'] and sha(E/'makefile.linux')==proof['make_sha256']
print('Source tree ready; next run prepare_inputs.py, run_official.py, inspect_binary_ports.py.')
