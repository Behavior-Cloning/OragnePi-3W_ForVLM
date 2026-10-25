"""Eliminate capacity-sized K/V concat by distributing the two matmuls.

Q @ cat(Kpast,Knew).T = cat(Q@Kpast.T,Q@Knew.T)
P @ cat(Vpast,Vnew) = Ppast@Vpast + Pnew@Vnew
Softmax still operates on all past/current scores with the original mask.
RoPE positions do not change. KV formats stay unchanged by default; the
explicit key_cache_transposed experiment requires a matching runtime layout.
"""
import copy
import numpy as np


def split_kv_attention(graph, quant, data, heads, kv, hd, capacity, tokens, *,
                       side='both', probability_i16=False, value_heads=False,
                       value_fp16=False, value_transposed=False, probability_identity=False,
                       softmax_i16=False, current_matmul=False, key_cache_transposed=False):
    assert tokens >= 1 and (tokens == 1 or current_matmul), 'batched current V requires matmul'
    nodes = graph['Layers']
    q = quant['quantize_parameters']
    node_id = lambda ref: ref[1:].rsplit(':', 1)[0]
    soft = next(k for k, n in nodes.items() if n['op'] == 'softmax')
    mm = [k for k, n in nodes.items() if n['op'] == 'matmul']
    assert len(mm) == 2
    av = next(k for k in mm if nodes[k]['inputs'][0] == '@' + soft + ':out0')
    qk = next(k for k in mm if k != av)
    trans = node_id(nodes[qk]['inputs'][1])
    assert nodes[trans]['op'] == 'permute' and nodes[trans]['parameters']['perm'] == [0, 1, 3, 2]
    kc = node_id(nodes[trans]['inputs'][0])
    vc = node_id(nodes[av]['inputs'][1])
    assert all(nodes[k]['op'] == 'concat' and nodes[k]['parameters']['dim'] == 2 for k in (kc, vc))
    past_k, new_k = nodes[kc]['inputs']
    past_v, new_v = nodes[vc]['inputs']
    def signature(ref):
        return {k: v for k, v in q[ref].items() if k not in ('min_value', 'max_value')}
    for concat, inputs in ((kc, (past_k, new_k)), (vc, (past_v, new_v))):
        assert all(signature(ref) == signature('@' + concat + ':out0') for ref in inputs), 'KV concat changes precision'
    def add(name, op, inputs, params, quant_ref):
        key = name + '_' + str(11000 + len(added))
        assert key not in nodes
        nodes[key] = dict(name=name, op=op, inputs=inputs, parameters=params, outputs=['out0'])
        q['@' + key + ':out0'] = copy.deepcopy(q[quant_ref])
        added.append(key)
        return '@' + key + ':out0'
    added = []
    query = nodes[qk]['inputs'][0]
    if key_cache_transposed:
        assert side != 'value'
        past = nodes[node_id(past_k)]
        assert past['op'] == 'input' and past['parameters']['shape'] == [1, kv, capacity, hd]
        past['parameters']['shape'] = [1, kv, hd, capacity]
        past_t = past_k
    else:
        past_t = add('split_kpast_t', 'permute', [past_k], {'perm': [0, 1, 3, 2]}, past_k)
    matparams = {'transpose_a': False, 'transpose_b': False}
    old_scores = add('split_scores_past', 'matmul', [query, past_t], matparams, '@' + qk + ':out0')
    # Keep the already qualified current-token QK reduction. Current AV has
    # a separate matmul candidate to avoid the SDK's cross-axis mul bug.
    fp16_ref = '@' + soft + ':out0'
    if tokens == 1:
        prod = add('split_scores_product', 'multiply', [query, new_k], {}, fp16_ref)
        mean = add('split_scores_mean', 'reducemean', [prod], {'axis_list': [-1], 'keep_dims': True}, fp16_ref)
        dim = add('split_head_dim', 'variable', [], {'shape': [1, 1, 1, 1]}, fp16_ref)
        data[node_id(dim)] = {'data': np.full((1, 1, 1, 1), hd, np.float32)}
        new_scores = add('split_scores_new', 'multiply', [mean, dim], {}, '@' + qk + ':out0')
    else:
        new_t = add('split_knew_t', 'permute', [new_k], {'perm': [0, 1, 3, 2]}, new_k)
        new_scores = add('split_scores_new', 'matmul', [query, new_t], matparams, '@' + qk + ':out0')
    if side != 'value':
        nodes[qk].update(op='concat', inputs=[old_scores, new_scores], parameters={'dim': 3})
    rows = heads // kv * tokens
    probs = '@' + soft + ':out0'
    old_probs = add('split_probs_past', 'slice', [probs],
                    {'begin': [0, 0, 0, 0], 'size': [1, kv, rows, capacity]}, probs)
    new_probs = add('split_probs_new', 'slice', [probs],
                    {'begin': [0, 0, 0, capacity], 'size': [1, kv, rows, tokens]}, probs)
    if probability_identity:
        # abs is exact for probabilities and gives the sliced tensor a
        # materializing producer without changing its floating precision.
        old_probs = add('split_probs_abs', 'abs', [old_probs], {}, probs)
    if probability_i16:
        old_probs = add('split_probs_contiguous', 'dtype_converter', [old_probs], {}, probs)
        q[old_probs] = {'qtype': 'i16', 'quantizer': 'dynamic_fixed_point', 'rounding': 'rtne', 'fl': 15}
    if value_fp16:
        past_v = add('split_value_fp16', 'dtype_converter', [past_v], {}, probs)
    if value_heads:
        values = []
        for head in range(kv):
            pr = add(f'split_prob_head{head}', 'slice', [old_probs],
                     {'begin': [0, head, 0, 0], 'size': [1, 1, rows, capacity]}, old_probs)
            vr = add(f'split_value_head{head}', 'slice', [past_v],
                     {'begin': [0, head, 0, 0], 'size': [1, 1, capacity, hd]}, past_v)
            pr = add(f'split_prob_matrix{head}', 'reshape', [pr], {'shape': [rows, capacity]}, old_probs)
            vr = add(f'split_value_matrix{head}', 'reshape', [vr], {'shape': [capacity, hd]}, past_v)
            out = add(f'split_value_mm{head}', 'matmul', [pr, vr], matparams, '@' + av + ':out0')
            values.append(add(f'split_value_restore{head}', 'reshape', [out], {'shape': [1, 1, rows, hd]}, out))
        old_value = add('split_value_past', 'concat', values, {'dim': 1}, '@' + av + ':out0')
    elif value_transposed:
        pr = add('split_probs_t', 'permute', [old_probs], {'perm': [0, 1, 3, 2]}, old_probs)
        vr = add('split_values_t', 'permute', [past_v], {'perm': [0, 1, 3, 2]}, past_v)
        vt = add('split_value_mm_t', 'matmul', [vr, pr], matparams, '@' + av + ':out0')
        old_value = add('split_value_past', 'permute', [vt], {'perm': [0, 1, 3, 2]}, '@' + av + ':out0')
    else:
        old_value = add('split_value_past', 'matmul', [old_probs, past_v], matparams, '@' + av + ':out0')
    new_value = add('split_value_new', 'matmul' if current_matmul else 'multiply',
                    [new_probs, new_v], matparams if current_matmul else {}, '@' + av + ':out0')
    if side != 'key':
        nodes[av].update(op='add', inputs=[old_value, new_value], parameters={})
    if softmax_i16:
        assert side != 'key' and not any((probability_i16, probability_identity, value_fp16, value_heads, value_transposed))
        pq = {'qtype': 'i16', 'quantizer': 'dynamic_fixed_point', 'rounding': 'rtne', 'fl': 15}
        for ref in (probs, old_probs, new_probs):q[ref] = pq.copy()
    # The old large concat and key transpose have no remaining consumers.
    live = set()
    def visit(k):
        if k in live:return
        live.add(k)
        for ref in nodes[k].get('inputs', []):visit(node_id(ref))
    for k, n in nodes.items():
        if n['op'] == 'output':visit(k)
    removed = sorted(set(nodes) - live)
    for k in removed:
        assert not any('@' + k + ':out0' in n.get('inputs', []) for key, n in nodes.items() if key not in removed)
        del nodes[k]
        del q['@' + k + ':out0']
        data.pop(k, None)
    return dict(strategy='split-past-current-kv-v3', side=side, key_cache_transposed=key_cache_transposed, current_matmul=current_matmul, softmax_i16=softmax_i16, probability_i16=probability_i16, probability_identity=probability_identity, value_heads=value_heads, value_fp16=value_fp16, value_transposed=value_transposed, added=[k for k in added if k in live], removed=removed,
                intermediate_precision='original score/AV quantizers; partial AV rounding must be validated')
