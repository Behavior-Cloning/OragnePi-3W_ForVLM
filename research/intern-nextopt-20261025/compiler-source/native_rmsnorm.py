"""Experimental SDK RMSNormalize replacement; not enabled by default.

Keep the learned scale multiply and all downstream quantizers unchanged. The
native operator uses unit scale, allowing the existing exported-weight oracle
to continue checking the learned norm constant. Target accumulator behaviour
must be established on the board, not inferred from the operator's name.
"""
import numpy as np


def replace_rmsnorm(graph, quant, data):
    nodes = graph['Layers']
    qp = quant['quantize_parameters']
    changes = []
    def key(ref):
        name, port = ref[1:].rsplit(':', 1)
        assert port == 'out0'
        return name
    for name in ('norm1', 'norm2'):
        scale = next(k for k, v in nodes.items() if v['name'] == name)
        consumers = [v for v in nodes.values() if '@' + scale + ':out0' in v.get('inputs', [])]
        assert len(consumers) == 1 and consumers[0]['op'] == 'multiply'
        root = next(key(v) for v in consumers[0]['inputs'] if key(v) != scale)
        node = nodes[root]
        assert node['op'] == 'multiply'
        divs = [nodes[key(v)] for v in node['inputs'] if nodes[key(v)]['op'] == 'real_div']
        assert len(divs) == 1, 'unknown stable norm decomposition'
        source, denominator = divs[0]['inputs']
        clip = nodes[key(denominator)]
        while clip['op'] == 'permute':
            clip = nodes[key(clip['inputs'][0])]
        assert clip['op'] == 'clipbyvalue'
        eps = float(clip['parameters']['clip_value_min']) ** 2
        assert 0 < eps <= 1e-3, 'requires stable norm v3 sqrt(eps) floor'
        assert qp['@' + root + ':out0'] == {'qtype': 'float16', 'quantizer': 'float16'}
        node.update(op='rmsnormalize', parameters={'axis': -1, 'eps': eps}, inputs=[source])
        width = data[scale]['data'].size
        data[root] = {'scale': np.ones(width, dtype=np.float32)}
        qp['@' + root + ':scale'] = {'qtype': 'float16', 'quantizer': 'float16'}
        changes.append({'norm': name, 'node': root, 'source': source, 'eps': eps,
                        'axis': -1, 'width': width, 'native_scale': 'unit FP16'})
    live = set()
    def retain(k):
        if k in live:
            return
        live.add(k)
        for ref in nodes[k].get('inputs', []):
            retain(key(ref))
    for k, v in nodes.items():
        if v['op'] == 'output':
            retain(k)
    removed = set(nodes) - live
    graph['Layers'] = {k: v for k, v in nodes.items() if k in live}
    quant['quantize_parameters'] = {k: v for k, v in qp.items() if k[1:].rsplit(':', 1)[0] in live}
    data = {k: v for k, v in data.items() if k in live}
    return data, {'norms': changes, 'removed_nodes': sorted(removed)}
