"""Explicit selective activation range changes, preserving all weight bytes."""


def widen_query_path(graph, quant, bits):
    """Widen Q projection/RoPE only, before compact/split attention rewriting."""
    assert 1<=bits<=4
    nodes=graph['Layers'];qp=quant['quantize_parameters']
    softmax=next(k for k,v in nodes.items() if v['op']=='softmax')
    matmuls=[k for k,v in nodes.items() if v['op']=='matmul'];assert len(matmuls)==2
    av=[k for k in matmuls if nodes[k]['inputs'][0]=='@'+softmax+':out0'];assert len(av)==1
    qk=next(k for k in matmuls if k!=av[0])
    pending=[nodes[qk]['inputs'][0][1:].rsplit(':',1)[0]]
    visited=set();changes=[];linears=set();boundaries=set()
    while pending:
        key=pending.pop()
        if key in visited:continue
        visited.add(key);node=nodes[key]
        if node['op']=='input':
            assert node['name'] in ('rope_cos','rope_sin'), node['name']
            boundaries.add(key);continue
        assert node['op'] in ('fullconnect','reshape','permute','slice','concat','neg','multiply','add','dtype_converter')
        ref='@'+key+':out0';old=qp[ref]
        if old['qtype']=='i16' and old['quantizer']=='dynamic_fixed_point':
            new={**old,'fl':old['fl']-bits};qp[ref]=new
            changes.append({'tensor':ref,'before':old,'after':new})
        else:
            assert old=={'qtype':'float16','quantizer':'float16'}, old
        if node['op']=='fullconnect':
            linears.add(key);continue  # Keep input Norm and W8/bias unchanged.
        pending.extend(ref[1:].rsplit(':',1)[0] for ref in node.get('inputs',[]))
    assert len(linears)==1 and len(boundaries)==2 and changes
    return {'headroom_bits':bits,'changes':changes,'query_projection':next(iter(linears)),
            'unchanged_rope_inputs':sorted(boundaries)}


def widen_mlp_product(graph, quant, bits):
    assert 1<=bits<=4
    nodes=graph['Layers'];qp=quant['quantize_parameters']
    swish=[k for k,v in nodes.items() if v['op']=='swish'];assert len(swish)==1
    products=[k for k,v in nodes.items() if v['op']=='multiply' and '@'+swish[0]+':out0' in v['inputs']]
    assert len(products)==1
    pending=products[:];seen=set();changes=[];linears=set()
    while pending:
        key=pending.pop()
        if key in seen:continue
        seen.add(key);node=nodes[key]
        assert key in products or node['op'] in ('reshape','permute','dtype_converter')
        ref='@'+key+':out0';old=qp[ref]
        assert old['qtype']=='i16' and old['quantizer']=='dynamic_fixed_point'
        new={**old,'fl':old['fl']-bits};qp[ref]=new
        changes.append({'tensor':ref,'before':old,'after':new})
        consumers=[k for k,v in nodes.items() if ref in v.get('inputs',[])]
        assert consumers
        for child in consumers:
            if nodes[child]['op']=='fullconnect':linears.add(child)
            else:pending.append(child)
    assert len(linears)==1,'expected exactly the MLP down projection'
    return {'headroom_bits':bits,'changes':changes,'down_projection':next(iter(linears))}
