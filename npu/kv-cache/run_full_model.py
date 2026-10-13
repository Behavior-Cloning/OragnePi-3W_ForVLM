"""Deploy and run every decoder layer plus NPU LM head with real token feedback."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
p.add_argument('--dtype',default='fp16');p.add_argument('--tag',required=True)
p.add_argument('--max-new',type=int,default=32);p.add_argument('--deploy-only',action='store_true')
p.add_argument('--skip-deploy',action='store_true',help='reuse deployed weights; still update prompt and runner')
p.add_argument('--case',help='unique result directory suffix, preserving earlier cases')
p.add_argument('--trace',action='store_true',help='diagnostic hidden states for all tokens/layers; not a clean speed run')
p.add_argument('--performance',action='store_true',help='no logits/hidden dumps or per-token logging; tokens and full-chain timing retained')
p.add_argument('--decoder-group',type=int,default=0,help='batch N decoder submissions and share FP16 hidden buffers; 0 keeps host transfers')
p.add_argument('--fusion-width',type=int,choices=(1,2,4),default=1,help='language layers per compiled decoder graph')
p.add_argument('--prefill-variant',help='compatible batched graph stack, loaded alongside single-token decode')
p.add_argument('--cached-prefix',type=int,default=0,help='prepare prefix KV before incremental TTFT timer starts')
p.add_argument('--prefill-scalar-prefix',type=int,default=0,help='keep first N absolute positions on the proven single-token path')
p.add_argument('--fast-greedy',action='store_true',help='compare shard integer maxima, checking against full FP32 argmax in diagnostic mode')
p.add_argument('--group-head',action='store_true',help='submit all complete output-head shards in one group, preserving per-shard quantization')
p.add_argument('--init-buffer-flush',action='store_true',help='diagnostic explicit initialization cache flush')
p.add_argument('--audit-outputs',action='store_true',help='diagnostic per-block output fingerprints; changes readback overhead')
p.add_argument('--capture-prefill',type=int,nargs=2,metavar=('POSITION','BLOCK'),help='capture one real prefill graph input/output for isolated replay')
p.add_argument('--capture-decode',type=int,nargs=2,metavar=('POSITION','BLOCK'),help='capture one real decode graph call; diagnostic ungrouped decoder only')
p.add_argument('--trace-decode-blocks',action='store_true',help='diagnostic complete-stack hidden at every fused block for scalar decode positions')
p.add_argument('--prefill-host-hidden',action='store_true',help='diagnostic host transfer between prefill blocks instead of aliasing')
p.add_argument('--host-output-buffers',action='store_true',help='diagnostic SDK host-backed cached output allocations')
p.add_argument('--prefill-rebind-outputs',action='store_true',help='diagnostic output rebinding before every batch call')
p.add_argument('--poison-prefill-output',action='store_true',help='diagnostic sentinel before device writes, not performance mode')
p.add_argument('--prefill-flush-output',action='store_true',help='flush hidden output before each prefill graph execution')
p.add_argument('--prefill-clear-output',action='store_true',help='clear and flush hidden output before each batched graph call')
p.add_argument('--prefill-scalar-block',type=int,default=-1,help='run one sensitive block through the single-token NPU graph during batch prefill')
p.add_argument('--cpu-affinity',help='taskset CPU list for the NPU host process')
p.add_argument('--feedback-from',type=Path,help='tokens.json of a frozen continuation; teacher forcing for matched work, not free generation')
p.add_argument('--reuse-tag',help='reuse immutable binaries from this deployed tag, hash-checking all reused files')
p.add_argument('--reuse-prefill',action='store_true',help='also hard-link unchanged prefill binaries from reuse-tag; SHA checks remain mandatory')
p.add_argument('--deploy-layers',type=int,nargs='+',default=[],help='new binaries to upload when --reuse-tag is used')
p.add_argument('--deploy-prefill-layers',type=int,nargs='+',default=[],help='upload selected changed prefill graphs while hard-linking the others')
p.add_argument('--prompt-folder',type=Path,help='case-specific prompt directory');args=p.parse_args()
from board_host_lock import lock_board_from_host
lock_board_from_host()
if not args.tag.replace('-','').isalnum():raise ValueError('invalid tag')
if args.trace and args.performance:raise ValueError('performance and trace are exclusive')
assert not args.trace_decode_blocks or not any((args.performance,args.trace,args.capture_decode,args.capture_prefill))
if args.capture_prefill:
    assert args.prefill_variant and not args.performance and min(args.capture_prefill)>=0
if args.capture_decode:
    assert not args.capture_prefill and not args.performance and not args.decoder_group
    assert min(args.capture_decode)>=0
assert not(args.poison_prefill_output and args.performance)
assert not(args.audit_outputs and args.performance)
if any((args.prefill_host_hidden,args.prefill_rebind_outputs,args.poison_prefill_output,args.prefill_flush_output,args.prefill_clear_output)):
    assert args.prefill_variant, 'prefill diagnostic options require a batch stack'
assert sum((args.poison_prefill_output,args.prefill_flush_output,args.prefill_clear_output))<=1
if args.cpu_affinity:assert 0<len(args.cpu_affinity)<=24 and all(c in '01234567,-' for c in args.cpu_affinity)
if args.reuse_tag and (not args.reuse_tag.replace('-','').isalnum() or args.reuse_tag==args.tag or args.skip_deploy):
    raise ValueError('invalid reuse tag or conflicting skip-deploy')
assert not args.reuse_prefill or (args.reuse_tag and args.prefill_variant)
assert not args.deploy_prefill_layers or args.reuse_prefill
if args.case and not args.case.replace('-','').isalnum():raise ValueError('invalid case')
manifest=json.loads((args.source/'trained-manifest.json').read_text())
count=len(manifest['layers'])
assert -1<=args.prefill_scalar_block<count//args.fusion_width
if args.prefill_scalar_block>=0:
    assert args.prefill_variant and not any((args.audit_outputs,args.capture_prefill,args.prefill_rebind_outputs,args.poison_prefill_output,args.prefill_flush_output,args.prefill_clear_output)), 'test scalar block independently of output diagnostics'
if count%args.fusion_width or not 0<=args.decoder_group<=count//args.fusion_width:raise ValueError('invalid fusion/group size')
if args.trace and args.fusion_width!=1:raise ValueError('fused graphs do not expose per-layer hidden traces')
records=json.loads((args.source/args.dtype/'stack-conversion.json').read_text())
key_cache_transposed = records[0].get('key_cache_transposed',False)
assert all(r.get('key_cache_transposed',False)==key_cache_transposed for r in records if r['layer']<count)
capacity=records[0].get('capacity',manifest['capacity'])
assert all(r.get('capacity',manifest['capacity'])==capacity and r.get('query_tokens',1)==1 for r in records if r['layer']<count)
prefill_records=[];prefill_tokens=0
if args.prefill_variant:
    assert not args.trace
    prefill_records=[r for r in json.loads((args.source/args.prefill_variant/'stack-conversion.json').read_text()) if r['layer']<count]
    prefill_tokens=prefill_records[0]['query_tokens']
    assert len(prefill_records)==count//args.fusion_width and prefill_tokens>1
    assert {r['layer'] for r in prefill_records}==set(range(0,count,args.fusion_width))
    assert all(r['query_tokens']==prefill_tokens and r['capacity']==capacity and r['fusion_width']==args.fusion_width and not r['exit_code'] for r in prefill_records)
    assert all(r.get('key_cache_transposed',False)==key_cache_transposed for r in prefill_records), 'prefill/decode K layouts must match'
head_shards=manifest.get('head_shards',[{'vocab_count':manifest['vocab_size']}])
assert len(records)==count//args.fusion_width+len(head_shards) and not any(r['exit_code'] for r in records)
assert {r['layer'] for r in records}==set(range(0,count,args.fusion_width))|set(range(count,count+len(head_shards)))
assert all(r.get('fusion_width',1)==args.fusion_width for r in records if r['layer']<count)
assert set(args.deploy_layers).issubset({r['layer'] for r in records}), 'unknown replacement layer'
assert set(args.deploy_prefill_layers).issubset({r['layer'] for r in prefill_records}), 'unknown replacement prefill layer'
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
prompt_count=len(json.loads((prompt_folder/'prompt.json').read_text())['token_ids'])
assert 0<=args.cached_prefix<prompt_count and 0<=args.prefill_scalar_prefix<=prompt_count
assert prompt_count+args.max_new-1<=capacity
(result_dir/'runner-config.json').write_text(json.dumps({'decode_variant':args.dtype,'prefill_variant':args.prefill_variant,
    'grouped_head':args.group_head,
    'key_cache_transposed':key_cache_transposed,
    'capacity':capacity,'query_tokens':prefill_tokens,'cached_prefix':args.cached_prefix,
    'scalar_prefix':args.prefill_scalar_prefix,'init_buffer_flush':args.init_buffer_flush,
    'audit_outputs':args.audit_outputs,'decoder_group':args.decoder_group,
    'capture_prefill':args.capture_prefill,'prefill_host_hidden':args.prefill_host_hidden,
    'capture_decode':args.capture_decode,
    'decode_block_trace':args.trace_decode_blocks,
    'host_output_buffers':args.host_output_buffers,'prefill_rebind_outputs':args.prefill_rebind_outputs,
    'poison_prefill_output':args.poison_prefill_output,'prefill_flush_output':args.prefill_flush_output,
    'prefill_clear_output':args.prefill_clear_output,'prefill_scalar_block':args.prefill_scalar_block,
    'host_cpu_affinity':args.cpu_affinity},indent=2)+'\n')
feedback_file=None
if args.feedback_from:
    feedback=json.loads(args.feedback_from.read_text())
    assert not feedback.get('fixed_feedback'), 'use a natural generation as the fixed trajectory'
    assert (args.feedback_from.parent/'prompt.i32').read_bytes()==(prompt_folder/'prompt.i32').read_bytes()
    assert len(feedback['tokens'])==args.max_new
    assert all(isinstance(t,int) and 0<=t<manifest['vocab_size'] for t in feedback['tokens'])
    feedback_file=result_dir/'feedback.i32'
    feedback_file.write_bytes(struct.pack('<'+'i'*args.max_new,*feedback['tokens']))
files=[(args.source/'embeddings.f32','embeddings.f32'),(prompt_folder/'prompt.i32',prompt_name),
       (scripts/'full_model.c','full_model.c'),(scripts/'board_probe.c','board_probe.c'),(scripts/'greedy_i16.h','greedy_i16.h'),
       (scripts/'batch_prefill.h','batch_prefill.h'),(scripts/'thermal_condition.py','thermal_condition.py'),(scripts/'thermal_heater.c','thermal_heater.c')]
files.extend((Path(r['path']),f"layer{r['layer']}.nb") for r in records)
files.extend((Path(r['path']),f"prefill-layer{r['layer']}.nb") for r in prefill_records)
if args.skip_deploy:
    files=[(src,name) for src,name in files if name in (prompt_name,'full_model.c','board_probe.c','greedy_i16.h','batch_prefill.h','thermal_condition.py','thermal_heater.c')]
if feedback_file:files.append((feedback_file,result_name+'-feedback.i32'))
if files:
    # Reserve room for both deployment and the complete diagnostic logits.
    # Hard-linked model files do not require another model-sized allocation.
    required=(0 if args.performance or args.deploy_only else args.max_new*manifest['vocab_size']*4)
    for src,name in files:
        linked=args.reuse_tag and (name=='embeddings.f32' or
            (args.reuse_prefill and name.startswith('prefill-layer') and name.endswith('.nb') and int(name[13:-3]) not in args.deploy_prefill_layers) or
            (name.startswith('layer') and name.endswith('.nb') and int(name[5:-3]) not in args.deploy_layers))
        if not linked:required+=src.stat().st_size
    available=int(subprocess.check_output(ssh+['python3 -c '+shlex.quote(
        'import shutil; print(shutil.disk_usage("/opt/opi-vlm/npu-kv-probe").free)')],text=True,timeout=20))
    if available<required+256*1024**2:
        raise RuntimeError(f'insufficient board space: {available} free, {required} plus 256 MiB reserve needed')
    subprocess.run(ssh+['mkdir -p '+remote],check=True,timeout=30)
    hashes=[]
    for source,name in files:
        reuse=args.reuse_tag and (name=='embeddings.f32' or
            (args.reuse_prefill and name.startswith('prefill-layer') and name.endswith('.nb') and int(name[13:-3]) not in args.deploy_prefill_layers) or
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
        f'-DFUSED_LAYERS={args.fusion_width} '
        f'-DKEY_CACHE_TRANSPOSED={int(key_cache_transposed)} '
        f'full_model.c -lNBGlinker -lVIPhal -lm -o {executable} && cc -O3 -Wall -Wextra -Werror -march=armv8-a+simd thermal_heater.c -pthread -o thermal_heater')
if args.deploy_only:
    subprocess.run(ssh+['flock -w 1800 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(compile_command)],check=True,timeout=1850)
    raise SystemExit(0)
for name in ('prompt.json','prompt.i32'):
    (result_dir/name).write_bytes((prompt_folder/name).read_bytes())
command=(f'cd {remote} && mkdir -p {result_name} && timeout 600 ./{executable} . {len(manifest["layers"])} {capacity} '
         f'{manifest["vocab_size"]} embeddings.f32 {prompt_name} {args.max_new} {result_name}/tokens.json {result_name}/logits.f32 {manifest["eos_token_ids"][0]}')
if args.trace:command=command.replace('timeout 600',f'OPI_FULL_TRACE={result_name}/hidden-trace.f32 timeout 600')
if args.trace_decode_blocks:command=command.replace('timeout 600',f'OPI_DECODE_BLOCK_TRACE={result_name}/decode-block-trace.f32 timeout 600')
if args.performance:command=command.replace('timeout 600','OPI_PERFORMANCE=1 timeout 600')
if args.decoder_group:command=command.replace('timeout 600',f'OPI_DECODER_GROUP={args.decoder_group} timeout 600')
if args.fast_greedy:command=command.replace('timeout 600','OPI_FAST_GREEDY=1 timeout 600')
if args.group_head:command=command.replace('timeout 600','OPI_GROUP_HEAD=1 timeout 600')
if args.init_buffer_flush:command=command.replace('timeout 600','OPI_INIT_BUFFER_FLUSH=1 timeout 600')
if args.host_output_buffers:command=command.replace('timeout 600','OPI_HOST_OUTPUT_BUFFERS=1 timeout 600')
if args.prefill_rebind_outputs:command=command.replace('timeout 600','OPI_PREFILL_REBIND_OUTPUTS=1 timeout 600')
if args.poison_prefill_output:command=command.replace('timeout 600','OPI_PREFILL_POISON_OUTPUT=1 timeout 600')
if args.prefill_flush_output:command=command.replace('timeout 600','OPI_PREFILL_FLUSH_OUTPUT=1 timeout 600')
if args.prefill_clear_output:command=command.replace('timeout 600','OPI_PREFILL_CLEAR_OUTPUT=1 timeout 600')
if args.prefill_scalar_block>=0:command=command.replace('timeout 600',f'OPI_PREFILL_SCALAR_BLOCK={args.prefill_scalar_block} timeout 600')
if args.audit_outputs:command=command.replace('timeout 600',f'OPI_AUDIT_OUTPUTS={result_name}/output-audit.tsv timeout 600')
if args.capture_prefill:
    position,block=args.capture_prefill
    assert block<count//args.fusion_width and position+prefill_tokens<=prompt_count
    command=command.replace('timeout 600',f'OPI_CAPTURE_POSITION={position} OPI_CAPTURE_BLOCK={block} OPI_CAPTURE_PREFIX={result_name}/capture timeout 600')
if args.capture_decode:
    position,block=args.capture_decode
    assert block<count//args.fusion_width and prompt_count<=position<prompt_count+args.max_new-1
    command=command.replace('timeout 600',f'OPI_DECODE_CAPTURE_POSITION={position} OPI_DECODE_CAPTURE_BLOCK={block} OPI_DECODE_CAPTURE_PREFIX={result_name}/decode-capture timeout 600')
if args.prefill_variant:command=command.replace('timeout 600',f'OPI_PREFILL_TOKENS={prefill_tokens} timeout 600')
if args.prefill_host_hidden:command=command.replace('timeout 600','OPI_PREFILL_HOST_HIDDEN=1 timeout 600')
if args.cached_prefix:command=command.replace('timeout 600',f'OPI_CACHED_PREFIX={args.cached_prefix} timeout 600')
if args.prefill_scalar_prefix:command=command.replace('timeout 600',f'OPI_PREFILL_SCALAR_PREFIX={args.prefill_scalar_prefix} timeout 600')
if feedback_file:command=command.replace('timeout 600',f'OPI_FEEDBACK_FILE={result_name}-feedback.i32 timeout 600')
if args.cpu_affinity:command=command.replace('timeout 600',f'taskset -c {args.cpu_affinity} timeout 600')
command=compile_command+' && '+command
command='flock -w 1800 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(command)
with (result_dir/'run.stdout').open('w') as out,(result_dir/'run.stderr').open('w') as err:
    result=subprocess.run(ssh+[command],stdout=out,stderr=err,timeout=2500)
(result_dir/'exit.json').write_text(json.dumps({'exit_code':result.returncode})+'\n')
if result.returncode:raise SystemExit(result.returncode)
for name in ('tokens.json',)+ (() if args.performance else ('logits.f32',))+ (('hidden-trace.f32',) if args.trace else ()) + (('decode-block-trace.f32',) if args.trace_decode_blocks else ()) + (('output-audit.tsv',) if args.audit_outputs else ()) + (('capture-input.f32','capture-output.f32') if args.capture_prefill else ()) + (('decode-capture-input.f32','decode-capture-output.f32') if args.capture_decode else ()):
    temporary=result_dir/(name+'.download')
    subprocess.run(scp+['root@192.168.88.29:'+remote+'/'+result_name+'/'+name,str(temporary)],check=True,timeout=120)
    temporary.replace(result_dir/name)
result=json.loads((result_dir/'tokens.json').read_text())
if result['prompt_tokens']!=len(json.loads((result_dir/'prompt.json').read_text())['token_ids']):
    raise ValueError('deployed prompt length mismatch')
print(json.dumps(result))
