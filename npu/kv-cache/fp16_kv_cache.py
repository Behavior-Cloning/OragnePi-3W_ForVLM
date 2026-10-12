"""Store KV as FP16 to avoid converting a full INT16 history every decode.

New K/V outputs retain their original quantizers. Runtime copy_values converts
only the newly appended rows. Q conversion also touches only the new token(s).
This is a different activation/cache configuration, never a weight-bit change.
"""
def fp16_kv_cache(graph, quant):
    nodes = graph['Layers']; q = quant['quantize_parameters']
    fp16 = {'qtype': 'float16', 'quantizer': 'float16'}
    changed = []; added = []
    def replace(ref):
        if q[ref] != fp16:
            changed.append({'tensor': ref, 'before': q[ref]})
            q[ref] = fp16.copy()
    def convert(ref, name):
        key = name + '_' + str(15000 + len(added))
        assert key not in nodes
        nodes[key] = {'name': name, 'op': 'dtype_converter', 'parameters': {},
                      'inputs': [ref], 'outputs': ['out0']}
        out = '@' + key + ':out0';q[out] = fp16.copy();added.append(key)
        return out
    for kind in ('k', 'v'):
        past = next(k for k, v in nodes.items() if v['name'] == 'past_' + kind)
        replace('@' + past + ':out0')
        concat = next(k for k, v in nodes.items() if v['op'] == 'concat' and
                      v['inputs'][0] == '@' + past + ':out0')
        nodes[concat]['inputs'][1] = convert(nodes[concat]['inputs'][1], 'cache_new_' + kind + '_fp16')
        replace('@' + concat + ':out0')
        if kind == 'k':
            transpose = next(k for k, v in nodes.items() if v['op'] == 'permute' and
                             v['inputs'] == ['@' + concat + ':out0'])
            replace('@' + transpose + ':out0')
            qk = next(v for v in nodes.values() if v['op'] == 'matmul' and
                      v['inputs'][1] == '@' + transpose + ':out0')
            qk['inputs'][0] = convert(qk['inputs'][0], 'cache_query_fp16')
    return {'strategy': 'fp16-resident-kv-v1', 'changed': changed, 'added': added,
            'external_new_kv_quantization_unchanged': True}
