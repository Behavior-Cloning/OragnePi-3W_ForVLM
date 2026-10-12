"""Read our own compiler-generated weight bytes for an independent FP32 oracle."""
from pathlib import Path
import hashlib
import re
import numpy as np
import torch

def read_constant(block, raw, arrays):
    match=re.search(r'VSI_NN_TYPE_(INT8|INT16|INT32|INT64|FLOAT16|FLOAT32),\s*(\d+),\s*(\d+)\)',block)
    if not match:raise ValueError('unsupported exported constant')
    dtype,offset,count=match.groups()
    npdtype=np.dtype({'INT8':'i1','INT16':'<i2','INT32':'<i4','INT64':'<i8','FLOAT16':'<f2','FLOAT32':'<f4'}[dtype])
    values=np.frombuffer(raw,npdtype,count=int(count)//npdtype.itemsize,offset=int(offset)).astype(np.float64)
    if dtype.startswith('INT'):
        scale_id=re.search(r'attr.dtype.scales = scales_(\d+)',block)
        if scale_id:
            if 'VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC' not in block:raise ValueError('unsupported affine constant')
            values*=arrays[scale_id[1]]
        else:
            fl=int(re.search(r'attr.dtype.fl = (-?\d+)',block)[1]);values*=2.**(-fl)
    return values.astype(np.float32),dtype

def apply_exported_int8(layer, folder: Path, prefix=''):
    graph=(folder/'export/vnn_decoder.c').read_text()
    arrays={k:np.asarray([float(v.strip()) for v in body.split(',') if v.strip()],np.float32)
            for k,body in re.findall(r'float\s+scales_(\d+)\[\]\s*=\s*\{([^}]+)\}',graph)}
    raw=(folder/'export/decoder.export.data').read_bytes()
    data=np.load(folder/'decoder.data',allow_pickle=True).item()
    records=[];matched=set()
    for name,parameters in data.items():
        if not name.startswith(prefix+'fullconnect'):continue
        original=parameters['weight']
        block=graph.split('/* @'+name+':weight */',1)[1].split('/*',1)[0]
        match=re.search(r'VSI_NN_TYPE_(INT8|FLOAT16),\s*(\d+),\s*(\d+)\)',block)
        if not match:raise ValueError('expected INT8 or explicit mixed FP16 decoder matrix')
        dtype=match[1];offset,count=map(int,match.groups()[1:])
        itemsize=1 if dtype=='INT8' else 2
        quantized=np.frombuffer(raw,np.int8 if itemsize==1 else np.float16,count=count//itemsize,offset=offset).reshape(original.shape[::-1]).T
        scale_id=re.search(r'attr.dtype.scales = scales_(\d+)',block)
        if dtype=='FLOAT16':scale=1.
        elif scale_id:
            scale=arrays[scale_id[1]][None,:]
        else:
            fl=int(re.search(r'attr.dtype.fl = (-?\d+)',block)[1]);scale=2.**(-fl)
        restored=quantized.astype(np.float32)*scale
        candidates=[key for key in ('q','k','v','o','gate','up','down')
                    if getattr(layer,key).shape==original.T.shape and np.array_equal(getattr(layer,key).numpy(),original.T)]
        assert len(candidates)==1,(name,candidates)
        key=candidates[0]
        matched.add(key)
        records.append({'matrix':key,'dtype':dtype,'relative_l2':float(np.linalg.norm(restored-original)/np.linalg.norm(original))})
        getattr(layer,key).copy_(torch.from_numpy(restored.T.copy()))
        bias_marker='/* @'+name+':bias */'
        target=getattr(layer,key+'_bias',None)
        if bias_marker not in graph:
            if target is not None or np.any(parameters.get('bias',0)):
                raise ValueError('missing nonzero exported bias '+name)
            continue
        bias_block=graph.split(bias_marker,1)[1].split('/*',1)[0]
        bias,bias_dtype=read_constant(bias_block,raw,arrays)
        if target is not None:
            original_bias=target.numpy()
            records.append({'matrix':key+'_bias','dtype':bias_dtype,'relative_l2':float(np.linalg.norm(bias-original_bias)/max(np.linalg.norm(original_bias),1e-20))})
            target.copy_(torch.from_numpy(bias.copy()))
        elif np.any(bias):raise ValueError('unexpected nonzero bias '+name)
    assert len(matched)==7
    for key in ('norm1','norm2','q_norm','k_norm'):
        target=getattr(layer,key,None)
        if target is None:continue
        blocks=[block for names,block in re.findall(r'/\* (.*?)\*/(.*?)(?=/\*|$)',graph,re.S)
                if re.search(r'@'+re.escape(prefix+key)+r'_\d+:from_var_data',names) and 'NEW_CONST_TENSOR' in block]
        if len(blocks)!=1:raise ValueError('ambiguous/missing norm constant '+key)
        restored,dtype=read_constant(blocks[0],raw,arrays)
        original=target.numpy()
        records.append({'matrix':key,'dtype':dtype,'relative_l2':float(np.linalg.norm(restored-original)/max(np.linalg.norm(original),1e-20))})
        target.copy_(torch.from_numpy(restored.copy()))
    return records


def apply_exported_head(head, source, dtype, manifest):
    """Reconstruct each deployed output-head shard, checking copied-head provenance."""
    records=[];norm=None
    shards=manifest.get('head_shards',[{'layer':manifest['language_layers_total'],
        'vocab_start':0,'vocab_count':manifest['vocab_size']}])
    for shard in shards:
        folder=source/dtype/f'l{shard["layer"]}'
        if not (folder/'export/vnn_decoder.c').exists():
            candidate=source/'fp16'/folder.name;binary='export_nbg_unify/network_binary.nb'
            if hashlib.sha256((candidate/binary).read_bytes()).digest()!=hashlib.sha256((folder/binary).read_bytes()).digest():
                raise ValueError('copied head provenance mismatch')
            folder=candidate
        graph=(folder/'export/vnn_decoder.c').read_text();raw=(folder/'export/decoder.export.data').read_bytes()
        arrays={k:np.asarray([float(v.strip()) for v in body.split(',') if v.strip()],np.float32)
                for k,body in re.findall(r'float\s+scales_(\d+)\[\]\s*=\s*\{([^}]+)\}',graph)}
        data=np.load(folder/'decoder.data',allow_pickle=True).item()
        names=[key for key in data if key.startswith('fullconnect')]
        if len(names)!=1:raise ValueError('expected one head matrix per shard')
        name=names[0];block=graph.split('/* @'+name+':weight */',1)[1].split('/*',1)[0]
        match=re.search(r'VSI_NN_TYPE_(INT8|INT16|FLOAT16),\s*(\d+),\s*(\d+)\)',block)
        if not match:raise ValueError('unsupported head weight dtype')
        kind,offset,count=match.groups();npdtype=np.dtype({'INT8':'i1','INT16':'<i2','FLOAT16':'<f2'}[kind])
        restored=np.frombuffer(raw,npdtype,count=int(count)//npdtype.itemsize,offset=int(offset)).astype(np.float32)
        start=shard['vocab_start'];end=start+shard['vocab_count'];target=head.weight[start:end]
        restored=restored.reshape(target.shape)
        if kind.startswith('INT'):
            scale_id=re.search(r'attr.dtype.scales = scales_(\d+)',block)
            if scale_id:restored*=arrays[scale_id[1]][:,None]
            else:restored*=2.**(-int(re.search(r'attr.dtype.fl = (-?\d+)',block)[1]))
        records.append({'matrix':f'head[{start}:{end}]','dtype':kind,
            'relative_l2':float(np.linalg.norm(restored-target.numpy())/np.linalg.norm(target.numpy()))})
        target.copy_(torch.from_numpy(restored.copy()))
        bias_block=graph.split('/* @'+name+':bias */',1)[1].split('/*',1)[0]
        bias,_=read_constant(bias_block,raw,arrays)
        if np.any(bias):raise ValueError('unexpected nonzero head bias')
        blocks=[block for labels,block in re.findall(r'/\* (.*?)\*/(.*?)(?=/\*|$)',graph,re.S)
                if re.search(r'@scale_\d+:from_var_data',labels) and 'NEW_CONST_TENSOR' in block]
        if len(blocks)!=1:raise ValueError('missing head norm scale')
        current,norm_kind=read_constant(blocks[0],raw,arrays)
        if norm is not None and not np.array_equal(norm,current):raise ValueError('sharded norm mismatch')
        norm=current
    records.append({'matrix':'output_norm','dtype':norm_kind,
        'relative_l2':float(np.linalg.norm(norm-head.scale.numpy())/np.linalg.norm(head.scale.numpy()))})
    head.scale.copy_(torch.from_numpy(norm.copy()))
    return records
