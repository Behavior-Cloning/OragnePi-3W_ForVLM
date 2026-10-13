"""Explicit, local W8A8 experiments without changing weights or KV ports.

The optional range reduction is declared in the conversion record. It is not
recalibration or proof of conversational quality; full-model gates still apply.
"""
import copy
import math
import numpy as np


def convert_fc_weights_dfp(graph, quant, data, hidden_dim, selection, method='peak', max_extra_bits=6):
    """Compare per-channel vs one-scale signed W8; activations are unchanged."""
    assert selection in ('gateup', 'mlp', 'all-w8') and method in ('peak', 'mse')
    assert type(max_extra_bits) is int and 0<=max_extra_bits<=6
    qp=quant['quantize_parameters'];changes=[]
    for name,node in graph['Layers'].items():
        if node['op']!='fullconnect':continue
        ref='@'+name+':weight';previous=qp[ref];matrix=data[name]['weight']
        if previous.get('qtype')!='i8' or previous.get('quantizer')=='dynamic_fixed_point':continue
        if selection=='gateup' and not (matrix.shape[0]==hidden_dim and matrix.shape[1]>hidden_dim):continue
        if selection=='mlp' and not ((matrix.shape[0]==hidden_dim and matrix.shape[1]>hidden_dim) or
                                    (matrix.shape[1]==hidden_dim and matrix.shape[0]>hidden_dim)):continue
        peak=float(np.abs(matrix).max());assert math.isfinite(peak) and peak>0
        peak_fl=math.floor(math.log2(127/peak))
        # Weight-only search: no prompts, continuation or holdout are consulted.
        # A few outlier weights can waste most of a max-range INT8 grid.
        # Retain the old max-range candidate and report every clipping tradeoff.
        trials=[]
        for fl in ([peak_fl] if method=='peak' else range(peak_fl,peak_fl+max_extra_bits+1)):
            integer=np.rint(matrix*2.**fl)
            restored=np.clip(integer,-128,127)*2.**(-fl)
            delta=matrix.astype(np.float64)-restored
            trials.append({'fl':fl,'mse':float(np.mean(delta*delta)),
                           'clipped_fraction':float(np.mean((integer < -128)|(integer > 127)))})
        best=min(trials,key=lambda t:t['mse'])
        candidate={'qtype':'i8','quantizer':'dynamic_fixed_point','rounding':'rtne',
                   'fl':best['fl'],
                   'max_value':float(matrix.max()),'min_value':float(matrix.min())}
        qp[ref]=candidate
        restored=np.clip(np.rint(matrix*2.**candidate['fl']),-128,127)*2.**(-candidate['fl'])
        changes.append({'linear':name,'before':previous,'after':candidate,
                        'scale_search':trials,'clipped_fraction':best['clipped_fraction'],
                        'weight_relative_l2':float(np.linalg.norm(matrix-restored)/np.linalg.norm(matrix))})
    assert changes
    return {'selection':selection,'method':method,'max_extra_bits':max_extra_bits,'changes':changes}


def convert_fc_inputs(graph, quant, data, hidden_dim, selection, range_bits=0):
    assert selection in ('gateup', 'all-w8') and range_bits in (0, 1)
    nodes = graph['Layers']
    qp = quant['quantize_parameters']
    converted, changes = {}, []
    for name, node in list(nodes.items()):
        if node['op'] != 'fullconnect':
            continue
        weight = qp['@' + name + ':weight']
        if weight.get('qtype') != 'i8':
            continue  # Preserve explicitly sensitive FP16 projections.
        matrix = data[name]['weight']
        if selection == 'gateup' and not (matrix.shape[0] == hidden_dim and matrix.shape[1] > hidden_dim):
            continue
        assert len(node['inputs']) == 1
        ref = node['inputs'][0]
        previous = qp[ref]
        assert previous['qtype'] == 'i16' and previous['quantizer'] == 'dynamic_fixed_point'
        # Preserve the existing range by default; a one-bit reduction is an
        # explicitly separate candidate, never silently applied to a baseline.
        candidate = copy.deepcopy(previous)
        candidate.update(qtype='i8', fl=previous['fl'] - 8 + range_bits)
        if ref not in converted:
            key = 'fc_input_i8_' + str(len(converted))
            assert key not in nodes
            nodes[key] = {'name': key, 'op': 'dtype_converter', 'parameters': {},
                          'inputs': [ref], 'outputs': ['out0']}
            converted[ref] = '@' + key + ':out0'
            qp[converted[ref]] = candidate
        else:
            assert qp[converted[ref]] == candidate
        node['inputs'][0] = converted[ref]
        changes.append({'linear': name, 'source': ref, 'converted': converted[ref],
                        'before': previous, 'after': candidate,
                        'weight_shape': list(matrix.shape)})
    assert changes, 'no eligible calibrated W8A16 matrices'
    return {'selection': selection, 'range_reduction_bits': range_bits, 'changes': changes}
