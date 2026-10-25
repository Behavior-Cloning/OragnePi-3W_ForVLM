"""Resize calibrated single-token ACUITY decoders without recalibration.

Compact GQA folds query heads and token rows together. The caller supplies a
causal mask repeated for each query head in a KV group, never repeats K/V.
"""
import copy
from compact_gqa import compact_gqa


def reshape_decoder(graph, quant, data, heads, kv, hd, capacity, tokens):
    assert 1 <= tokens <= capacity <= 8192 and heads % kv == 0
    nodes=graph['Layers']
    hidden=next(v for v in nodes.values() if v['name']=='hidden')
    past=next(v for v in nodes.values() if v['name']=='past_k')
    old_capacity=past['parameters']['shape'][2]
    assert hidden['parameters']['shape']==[1,1,heads*hd]
    data, layout=compact_gqa(graph,quant,data,heads,kv,hd)
    nodes=graph['Layers'];q=quant['quantize_parameters'];repeat=heads//kv
    shapes={'hidden':[1,tokens,heads*hd], 'past_k':[1,kv,capacity,hd],
            'past_v':[1,kv,capacity,hd], 'mask':[1,1,repeat*tokens,capacity+tokens],
            'rope_cos':[1,1,tokens,hd], 'rope_sin':[1,1,tokens,hd]}
    # Preserve the proven single-token mask broadcast contract.
    if tokens==1:shapes['mask']=[1,1,1,capacity+1]
    changed=[]
    for name,node in nodes.items():
        before=copy.deepcopy(node.get('parameters',{}));params=node.get('parameters',{})
        if node['op']=='input':
            params['shape']=shapes[node['name']]
        elif node['op']=='reshape':
            shape=params['shape']
            if node['name']=='compact_query':params['shape']=[1,kv,repeat*tokens,hd]
            elif len(shape)==3 and shape[0:2]==[-1,1]:params['shape']=[1,tokens,shape[-1]]
            elif node['name']=='Transpose_4_output_0':
                # compact_gqa changed the old AV permute to a flat reshape.
                assert shape==[1,1,heads*hd]
                params['shape']=[1,heads,tokens,hd]
        elif node['op']=='slice':
            assert params['size'][2]==1 and params['begin'][2]==0
            params['size'][2]=tokens
        if before!=params:changed.append({'node':name,'before':before,'after':copy.deepcopy(params)})
    flatten=next(k for k,v in nodes.items() if v['name']=='Transpose_4_output_0')
    consumers=[v for v in nodes.values() if '@'+flatten+':out0' in v.get('inputs',[])]
    assert len(consumers)==1
    name='batch_attention_permute_10001'
    nodes[name]={'name':'batch_attention_permute','op':'permute','parameters':{'perm':[0,2,1,3]},
                 'inputs':['@'+flatten+':out0'],'outputs':['out0']}
    q['@'+name+':out0']=copy.deepcopy(q['@'+flatten+':out0'])
    consumers[0]['inputs']=['@'+name+':out0' if v=='@'+flatten+':out0' else v for v in consumers[0]['inputs']]
    return data, {'old_capacity':old_capacity,'capacity':capacity,'tokens':tokens,
                  'mask_query_repeats':repeat if tokens>1 else 1,'layout':layout,'changes':changed}


def slice_value_layout(graph,quant,kv,hd,tokens):
    """Materialize [KV,T,D] using head slices instead of a batched permute."""
    nodes=graph['Layers'];q=quant['quantize_parameters']
    key=next(k for k,v in nodes.items() if v['name']=='new_v')
    node=nodes[key];assert node['op']=='permute' and node['parameters']['perm']==[0,2,1,3]
    source=node['inputs'][0];inputs=[]
    for h in range(kv):
        cut=f'value_slice_{11000+h}';flat=f'value_slice_reshape_{12000+h}'
        nodes[cut]={'name':cut,'op':'slice','parameters':{'begin':[0,0,h,0],'size':[1,tokens,1,hd]},
                    'inputs':[source],'outputs':['out0']}
        nodes[flat]={'name':flat,'op':'reshape','parameters':{'shape':[1,1,tokens,hd]},
                    'inputs':['@'+cut+':out0'],'outputs':['out0']}
        for new in (cut,flat):q['@'+new+':out0']=copy.deepcopy(q['@'+key+':out0'])
        inputs.append('@'+flat+':out0')
    node.update(op='concat',parameters={'dim':1},inputs=inputs)
