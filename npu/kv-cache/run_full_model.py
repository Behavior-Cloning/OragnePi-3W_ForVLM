"""Deploy and run every decoder layer plus NPU LM head with real token feedback."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
p.add_argument('--dtype',default='fp16');p.add_argument('--tag',required=True)
p.add_argument('--max-new',type=int,default=32);p.add_argument('--deploy-only',action='store_true')
p.add_argument('--skip-deploy',action='store_true',help='reuse deployed weights; still update prompt and runner')
p.add_argument('--case',help='unique result directory suffix, preserving earlier cases')
p.add_argument('--trace',action='store_true',help='diagnostic hidden states for all tokens/layers; not a clean speed run')
p.add_argument('--performance',action='store_true',help='no logits/hidden dumps or per-token logging; tokens and full-chain timing retained')
p.add_argument('--decoder-group',type=int,default=0,help='batch N decoder submissions and share FP16 hidden buffers; 0 keeps host transfers')
p.add_argument('--reuse-tag',help='reuse immutable binaries from this deployed tag, hash-checking all reused files')
p.add_argument('--deploy-layers',type=int,nargs='+',default=[],help='new binaries to upload when --reuse-tag is used')
p.add_argument('--prompt-folder',type=Path,help='case-specific prompt directory');args=p.parse_args()
if not args.tag.replace('-','').isalnum():raise ValueError('invalid tag')
if args.trace and args.performance:raise ValueError('performance and trace are exclusive')
if args.reuse_tag and (not args.reuse_tag.replace('-','').isalnum() or args.reuse_tag==args.tag or args.skip_deploy):
    raise ValueError('invalid reuse tag or conflicting skip-deploy')
if args.case and not args.case.replace('-','').isalnum():raise ValueError('invalid case')
manifest=json.loads((args.source/'trained-manifest.json').read_text())
if not 0<=args.decoder_group<=len(manifest['layers']):raise ValueError('invalid decoder group size')
records=json.loads((args.source/args.dtype/'stack-conversion.json').read_text())
head_shards=manifest.get('head_shards',[{'vocab_count':manifest['vocab_size']}])
assert len(records)==len(manifest['layers'])+len(head_shards) and not any(r['exit_code'] for r in records)
assert set(args.deploy_layers).issubset({r['layer'] for r in records}), 'unknown replacement layer'
remote='/opt/opi-vlm/npu-kv-probe/'+args.tag
prefix=[str(Path.home()/'.local/bin/sshpass'),'-e'] if 'SSHPASS' in os.environ else []
ssh=prefix+['ssh','-o','ForwardX11=no','-o','ConnectTimeout=8','root@192.168.88.29']
scp=prefix+['scp','-o','ForwardX11=no','-o','ConnectTimeout=8']
scripts=Path(__file__).resolve().parent
result_name='board'+('-'+args.case if args.case else '')
prompt_name=result_name+'-prompt.i32'
executable=result_name+'-full_model'
result_dir=args.source/args.dtype/result_name;result_dir.mkdir(exist_ok=True)
prompt_folder=args.prompt_folder or args.source
files=[(args.source/'embeddings.f32','embeddings.f32'),(prompt_folder/'prompt.i32',prompt_name),
       (scripts/'full_model.c','full_model.c'),(scripts/'board_probe.c','board_probe.c')]
files.extend((Path(r['path']),f"layer{r['layer']}.nb") for r in records)
if args.skip_deploy:
    files=[(src,name) for src,name in files if name in (prompt_name,'full_model.c','board_probe.c')]
if files:
    subprocess.run(ssh+['mkdir -p '+remote],check=True,timeout=30)
    hashes=[]
    for source,name in files:
        reuse=args.reuse_tag and (name=='embeddings.f32' or
            (name.startswith('layer') and name.endswith('.nb') and int(name[5:-3]) not in args.deploy_layers))
        if reuse:
            original='/opt/opi-vlm/npu-kv-probe/'+args.reuse_tag+'/'+name
            # Hard links make this variant independent of later path replacement.
            subprocess.run(ssh+['ln '+shlex.quote(original)+' '+shlex.quote(remote+'/'+name)],check=True,timeout=30)
        else:
            subprocess.run(scp+[str(source),'root@192.168.88.29:'+remote+'/'+name],check=True,timeout=600)
        hashes.append(hashlib.file_digest(source.open('rb'),'sha256').hexdigest()+'  '+name)
        print('uploaded',name,flush=True)
    verification='cd '+remote+' && sha256sum -c -'
    subprocess.run(ssh+[verification],input='\n'.join(hashes)+'\n',text=True,check=True,timeout=120)
    (result_dir/'deployed-sha256.txt').write_text('\n'.join(hashes)+'\n')
    compile_command=(f'cd {remote} && cc -O2 -Wall -Wextra -Werror -DMODEL_DIM={manifest["dim"]} '
        f'-DMODEL_KV={manifest["kv_heads"]} -DMODEL_HD={manifest["head_dim"]} -DROPE_BASE={manifest["rope_base"]} '
        f'-DHEAD_SHARDS={len(head_shards)} -DHEAD_CHUNK={head_shards[0]["vocab_count"]} '
        f'full_model.c -lNBGlinker -lVIPhal -lm -o {executable}')
if args.deploy_only:
    subprocess.run(ssh+['flock -w 1800 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(compile_command)],check=True,timeout=1850)
    raise SystemExit(0)
for name in ('prompt.json','prompt.i32'):
    (result_dir/name).write_bytes((prompt_folder/name).read_bytes())
command=(f'cd {remote} && mkdir -p {result_name} && timeout 600 ./{executable} . {len(manifest["layers"])} {manifest["capacity"]} '
         f'{manifest["vocab_size"]} embeddings.f32 {prompt_name} {args.max_new} {result_name}/tokens.json {result_name}/logits.f32 {manifest["eos_token_ids"][0]}')
if args.trace:command=command.replace('timeout 600',f'OPI_FULL_TRACE={result_name}/hidden-trace.f32 timeout 600')
if args.performance:command=command.replace('timeout 600','OPI_PERFORMANCE=1 timeout 600')
if args.decoder_group:command=command.replace('timeout 600',f'OPI_DECODER_GROUP={args.decoder_group} timeout 600')
command=compile_command+' && '+command
command='flock -w 1800 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(command)
with (result_dir/'run.stdout').open('w') as out,(result_dir/'run.stderr').open('w') as err:
    result=subprocess.run(ssh+[command],stdout=out,stderr=err,timeout=2500)
(result_dir/'exit.json').write_text(json.dumps({'exit_code':result.returncode})+'\n')
if result.returncode:raise SystemExit(result.returncode)
for name in ('tokens.json',)+ (() if args.performance else ('logits.f32',))+ (('hidden-trace.f32',) if args.trace else ()):
    subprocess.run(scp+['root@192.168.88.29:'+remote+'/'+result_name+'/'+name,str(result_dir/name)],check=True,timeout=120)
result=json.loads((result_dir/'tokens.json').read_text())
if result['prompt_tokens']!=len(json.loads((result_dir/'prompt.json').read_text())['token_ids']):
    raise ValueError('deployed prompt length mismatch')
print(json.dumps(result))
