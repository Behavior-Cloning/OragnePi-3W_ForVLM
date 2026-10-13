"""Explicit selective activation range changes, preserving all weight bytes."""


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
