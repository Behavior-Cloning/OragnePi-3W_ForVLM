"""Bounded weight-only INT8 export; successful export alone is not acceleration."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import uuid
import numpy as np
import yaml

p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--timeout',type=int,default=180)
p.add_argument('--weight-mode',choices=['perchannel','dfp'],default='perchannel')
p.add_argument('--expected-linears',type=int,default=7)
p.add_argument('--only-linear',nargs='+',help='quantize only these imported linear layer names')
p.add_argument('--dfp-linear',nargs='+',default=[],help='use DFP INT8 for these names, retaining per-channel INT8 elsewhere')
p.add_argument('--value-dfp',action='store_true',help='resolve the V projection by graph dependencies and use DFP INT8')
p.add_argument('--float-nonlinear',action='store_true',help='retain INT16 linear boundaries but use FP16 for nonlinear graph tensors')
p.add_argument('--float-residual',action='store_true',help='keep decoder input/output and residual sums in FP16')
p.add_argument('--float-sensitive',action='store_true',help='FP16 RMSNorm internals and masked softmax; keep INT16 linear boundaries')
p.add_argument('--activation-headroom-bits',type=int,default=0,help='diagnostic expansion of INT16 activation ranges, excluding weights/constants')
p.add_argument('--float-down-output',action='store_true',help='FP16 down-projection output to retain ordinary values next to BOS outliers')
p.add_argument('--float-mlp',action='store_true',help='FP16 MLP only, for layers with exceptional BOS activation outliers')
args=p.parse_args();root=Path(__file__).resolve().parents[2]/'npu/results'
base=args.base.resolve();out=args.output.resolve();relative=out.relative_to(root)
out.mkdir(parents=True,exist_ok=True)
for name in ('decoder.json','decoder.data','tensor-inputmeta.yml'):
    shutil.copyfile(base/name,out/name)
quant=yaml.safe_load((base/'decoder.quantize').read_text())
graph=json.loads((base/'decoder.json').read_text())['Layers']
if args.float_residual:
    names={k for k,v in graph.items() if v.get('name') in ('hidden','hidden_out','Add_4_output_0','attach_hidden_out/out0')}
    assert len(names)==4,names
    for name in names:
        key='@'+name+':out0'
        quant['quantize_parameters'][key]={'qtype':'float16','quantizer':'float16'}
if args.float_sensitive:
    # Keep the final normalized outputs INT16 for the W8A16 linear kernels.
    # Traverse only the two normalization subgraphs, stopping at residuals.
    terminals={k for k,v in graph.items() if v.get('name') in ('hidden','Add_4_output_0')}
    roots=[k for k,v in graph.items() if v.get('name') in ('Mul_4_output_0','Mul_13_output_0')]
    assert len(terminals)==2 and len(roots)==2, 'unsupported normalization graph'
    floating=set()
    def visit_norm(name):
        if name in floating or name in terminals:return
        assert graph[name]['op']!='fullconnect', 'normalization traversal escaped its boundary'
        floating.add(name)
        for value in graph[name].get('inputs',[]):
            if value.startswith('@'):visit_norm(value[1:].split(':')[0])
    for root_name in roots:
        for value in graph[root_name]['inputs']:
            if value.startswith('@'):visit_norm(value[1:].split(':')[0])
    floating.update(k for k,v in graph.items() if v.get('name') in ('Add_3_output_0','Softmax_output_0'))
    for name in floating:
        quant['quantize_parameters']['@'+name+':out0']={'qtype':'float16','quantizer':'float16'}
if args.value_dfp:
    values=[k for k,v in graph.items() if v.get('name')=='new_v']
    assert len(values)==1, 'expected a unique new_v graph node'
    def linear_ancestors(name):
        if graph[name]['op']=='fullconnect':return {name}
        return set().union(*(linear_ancestors(v[1:].split(':')[0]) for v in graph[name].get('inputs',[]) if v.startswith('@')))
    value_linear=linear_ancestors(values[0]);assert len(value_linear)==1,value_linear
    args.dfp_linear+=sorted(value_linear)
if args.float_nonlinear:
    preserve=set()
    for name,info in graph.items():
        if info['op']=='fullconnect':
            preserve.update(info['inputs']);preserve.add('@'+name+':out0')
    for key in list(quant['quantize_parameters']):
        if key not in preserve and not key.endswith((':weight',':bias')):
            quant['quantize_parameters'][key]={'qtype':'float16','quantizer':'float16'}
if not 0<=args.activation_headroom_bits<=8:raise ValueError('activation headroom must be 0..8 bits')
if args.activation_headroom_bits:
    for key,q in quant['quantize_parameters'].items():
        node=key[1:].split(':')[0]
        if key.endswith(':out0') and q.get('qtype')=='i16' and graph.get(node,{}).get('op')!='variable':
            q['fl']-=args.activation_headroom_bits
if args.float_down_output:
    hidden_out=next(k for k,v in graph.items() if v.get('name')=='hidden_out')
    branches=[v[1:].split(':')[0] for v in graph[hidden_out]['inputs']]
    branch=next(k for k in branches if graph[k].get('name')!='Add_4_output_0')
    while True:
        quant['quantize_parameters']['@'+branch+':out0']={'qtype':'float16','quantizer':'float16'}
        if graph[branch]['op']=='fullconnect':break
        assert graph[branch]['op'] in ('reshape','permute'),graph[branch]
        branch=graph[branch]['inputs'][0][1:].split(':')[0]
float_linears=set()
if args.float_mlp:
    stop=next(k for k,v in graph.items() if v.get('name')=='Add_4_output_0')
    output=next(k for k,v in graph.items() if v.get('name')=='hidden_out')
    seen=set()
    def float_branch(node):
        if node==stop or node in seen:return
        seen.add(node)
        quant['quantize_parameters']['@'+node+':out0']={'qtype':'float16','quantizer':'float16'}
        if graph[node]['op']=='fullconnect':float_linears.add(node)
        for value in graph[node].get('inputs',[]):
            if value.startswith('@'):float_branch(value[1:].split(':')[0])
    float_branch(output)
    assert len(float_linears)==3,float_linears
# This is our own compiler-generated numpy dictionary, not an external pickle.
data=np.load(base/'decoder.data',allow_pickle=True).item();changed=[]
for name,parameters in data.items():
    key=f'@{name}:weight'
    if name.startswith('fullconnect') and key in quant['quantize_parameters']:
        if args.only_linear and name not in args.only_linear:continue
        weight=parameters['weight']
        if name in float_linears:
            quant['quantize_parameters'][key]={'qtype':'float16','quantizer':'float16'}
            # INT16 calibration left INT64 bias metadata. FP16 kernels need
            # the compiler's floating bias inference, as in a pure FP16 export.
            quant['quantize_parameters'].pop('@'+name+':bias',None)
            changed.append({'name':name,'shape':list(weight.shape),'dtype':'float16'})
            continue
        scales=np.maximum(np.max(np.abs(weight),axis=0)/127,np.finfo(np.float32).tiny)
        if args.weight_mode=='perchannel' and name not in args.dfp_linear:
            quant['quantize_parameters'][key]={'qtype':'i8','quantizer':'perchannel_symmetric_affine',
                'rounding':'rtne','scale':scales.tolist(),'channel_dim':-1}
        else:
            fl=int(np.floor(np.log2(127/max(float(np.abs(weight).max()),1e-12))))
            quant['quantize_parameters'][key]={'qtype':'i8','quantizer':'dynamic_fixed_point',
                'rounding':'rtne','fl':fl,'max_value':float(weight.max()),'min_value':float(weight.min())}
        changed.append({'name':name,'shape':list(weight.shape),'dtype':'int8'})
assert len(changed)==args.expected_linears, 'unexpected linear projection count'
(out/'decoder.quantize').write_text(yaml.safe_dump(quant,sort_keys=False))
report={'scope':'one decoder with explicit per-matrix weight and sensitive-activation precision overrides',
        'int8_matrices':sum(w['dtype']=='int8' for w in changed),
        'fp16_matrices':sum(w['dtype']=='float16' for w in changed),
        'base':str(base.relative_to(root)),'weights':changed,'weight_mode':args.weight_mode,'dfp_linear':args.dfp_linear,'float_nonlinear':args.float_nonlinear,'float_residual':args.float_residual,
        'float_sensitive':args.float_sensitive,
        'activation_headroom_bits':args.activation_headroom_bits,
        'float_down_output':args.float_down_output,
        'float_mlp':args.float_mlp,
        'source_quantize_sha256':hashlib.file_digest((base/'decoder.quantize').open('rb'),'sha256').hexdigest()}
container='opi-w8-test-'+uuid.uuid4().hex[:12]
command=['docker','run','--rm','--name',container,'--network','none','--cpus','4','--memory','8g',
    '-e','TF_CPP_MIN_LOG_LEVEL=3','-e','OMP_NUM_THREADS=4',
    '-e','VSIMULATOR_SHADER_CORE_COUNT=1','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B',
    '-v',str(root)+':/work','-w','/work/'+str(relative),'ubuntu-npu:v2.0.10.2',
    'python3','/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py','export','ovxlib',
    '--model','decoder.json','--model-data','decoder.data','--model-quantize','decoder.quantize',
    '--dtype','quantized','--with-input-meta','tensor-inputmeta.yml','--target-ide-project','linux64',
    '--pack-nbg-unify','--optimize','VIP9000NANODI_PLUS_PID0X1000003B',
    '--viv-sdk','/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools','--output-path','export/decoder']
start=time.monotonic()
with (out/'export.log').open('w') as log:
    try:
        proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
        report['exit_code']=proc.returncode
    except subprocess.TimeoutExpired:
        subprocess.run(['docker','stop','-t','2',container],stdout=log,stderr=subprocess.STDOUT,timeout=15)
        report['exit_code']=124
report['seconds']=time.monotonic()-start
report['binaries']=[{'bytes':f.stat().st_size,'path':str(f.relative_to(out))} for f in out.rglob('network_binary.nb')]
(out/'conversion.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
raise SystemExit(report['exit_code'])
