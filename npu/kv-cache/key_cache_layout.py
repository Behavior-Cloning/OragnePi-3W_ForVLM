"""Keep cached K in [batch, kv_head, head_dim, capacity] for Q @ K.

Only transpose the new query block, not the whole history at every step.
External new-K output remains [batch, kv_head, query_tokens, head_dim].
"""
def transpose_key_cache(graph, quant):
    nodes=graph['Layers'];q=quant['quantize_parameters']
    key=lambda ref:ref[1:].rsplit(':',1)[0]
    past=next(k for k,v in nodes.items() if v['name']=='past_k')
    new=next(k for k,v in nodes.items() if v['name']=='new_k')
    concat=next(k for k,v in nodes.items() if v['op']=='concat' and v['inputs']==['@'+past+':out0','@'+new+':out0'])
    permutes=[k for k,v in nodes.items() if v.get('inputs')==['@'+concat+':out0']]
    assert len(permutes)==1, 'requires compact GQA with no K repeat'
    transpose=permutes[0]
    assert nodes[transpose]['op']=='permute' and nodes[transpose]['parameters']['perm']==[0,1,3,2]
    consumers=[k for k,v in nodes.items() if '@'+transpose+':out0' in v.get('inputs',[])]
    assert len(consumers)==1 and nodes[consumers[0]]['op']=='matmul'
    def signature(k):
        return {a:b for a,b in q['@'+k+':out0'].items() if a not in ('min_value','max_value')}
    assert all(signature(k)==signature(past) for k in (new,concat,transpose)), 'moving transpose would move a quantization boundary'
    shape=nodes[past]['parameters']['shape'];assert len(shape)==4
    before=shape[:]
    nodes[past]['parameters']['shape']=[shape[0],shape[1],shape[3],shape[2]]
    nodes[transpose]['inputs']=['@'+new+':out0']
    nodes[concat]['inputs']=['@'+past+':out0','@'+transpose+':out0']
    nodes[concat]['parameters']['dim']=3
    nodes[consumers[0]]['inputs'][1]='@'+concat+':out0'
    return {'past_k_shape_before':before,'past_k_shape_after':nodes[past]['parameters']['shape'],
            'qk_node':consumers[0],'new_k_output_layout_unchanged':True,
            'quantization_boundaries_identical':True}
