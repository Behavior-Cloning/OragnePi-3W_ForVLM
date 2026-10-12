"""Use grouped Q rows instead of physically repeating K/V for each Q head."""
import copy

def compact_gqa(graph, quant, data, heads, kv, hd):
    assert heads % kv == 0
    nodes = graph['Layers']
    hidden = next(v for v in nodes.values() if v['name'] == 'hidden')
    assert hidden['parameters']['shape'] == [1, 1, heads * hd], 'single-token decode only'
    q = quant['quantize_parameters']
    matmuls = [k for k, v in nodes.items() if v['op'] == 'matmul']
    assert len(matmuls) == 2
    softmax = next(k for k, v in nodes.items() if v['op'] == 'softmax')
    av = next(k for k in matmuls if nodes[k]['inputs'][0] == '@' + softmax + ':out0')
    qk = next(k for k in matmuls if k != av)
    node_id = lambda ref: ref[1:].rsplit(':', 1)[0]
    def quant_signature(key):
        value = q['@' + key + ':out0']
        return {k: v for k, v in value.items() if k not in ('min_value', 'max_value')}
    def remove_repeat(last):
        # Imported repeat_interleave: concat -> unsqueeze -> tile -> reshape.
        reshape = nodes[last]
        tile = node_id(reshape['inputs'][0])
        unsqueeze = node_id(nodes[tile]['inputs'][0])
        concat = node_id(nodes[unsqueeze]['inputs'][0])
        assert reshape['op'] == nodes[unsqueeze]['op'] == 'reshape'
        assert nodes[tile]['op'] == 'tile' and nodes[concat]['op'] == 'concat'
        assert nodes[tile]['parameters']['multiples'] == [1, 1, heads // kv, 1, 1]
        assert all(quant_signature(k) == quant_signature(concat) for k in (last, tile, unsqueeze)), 'repeat path changes precision'
        return '@' + concat + ':out0'
    transpose = node_id(nodes[qk]['inputs'][1])
    assert nodes[transpose]['op'] == 'permute' and nodes[transpose]['parameters']['perm'] == [0, 1, 3, 2]
    nodes[transpose]['inputs'][0] = remove_repeat(node_id(nodes[transpose]['inputs'][0]))
    nodes[av]['inputs'][1] = remove_repeat(node_id(nodes[av]['inputs'][1]))
    query = nodes[qk]['inputs'][0]
    regroup = 'compact_query_10000'
    assert regroup not in nodes
    nodes[regroup] = {'name': 'compact_query', 'op': 'reshape',
        'parameters': {'shape': [1, kv, heads // kv, hd]}, 'inputs': [query], 'outputs': ['out0']}
    q['@' + regroup + ':out0'] = copy.deepcopy(q[query])
    nodes[qk]['inputs'][0] = '@' + regroup + ':out0'
    consumers = [k for k, v in nodes.items() if '@' + av + ':out0' in v.get('inputs', [])]
    assert len(consumers) == 1
    flatten = consumers[0]
    assert nodes[flatten]['op'] == 'permute' and nodes[flatten]['parameters']['perm'] == [0, 2, 1, 3]
    nodes[flatten]['op'] = 'reshape'
    nodes[flatten]['parameters'] = {'shape': [1, 1, heads * hd]}
    live = set()
    def visit(key):
        if key in live:
            return
        live.add(key)
        for ref in nodes[key].get('inputs', []):
            visit(node_id(ref))
    for key, value in nodes.items():
        if value['op'] == 'output':
            visit(key)
    removed = sorted(set(nodes) - live)
    graph['Layers'] = {k: v for k, v in nodes.items() if k in live}
    quant['quantize_parameters'] = {k: v for k, v in q.items() if node_id(k) in live}
    data = {k: v for k, v in data.items() if k in live}
    return data, {'removed_nodes': removed, 'query_shape': [1, kv, heads // kv, hd],
                  'kv_repeat_factor_removed': heads // kv}
