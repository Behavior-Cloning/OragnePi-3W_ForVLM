"""Read exported C port order; ACUITY sorts node IDs across merged graphs."""
import json
import re

def verify_contract(folder, width):
    graph = json.loads((folder / 'decoder.json').read_text())['Layers']
    code = (folder / 'export/vnn_decoder.c').read_text()
    tensors = {}
    for label, body in re.findall(r'/\* @([^*]+?) \*/(.*?)(?=/\*|$)', code, re.S):
        match = re.search(r'NEW_NORM_TENSOR\(norm_tensor\[(\d+)\]', body)
        if match:
            tensors[int(match[1])] = label.strip().rsplit(':', 1)[0]
    def port_order(kind):
        return [tensors[int(tensor)] for _, tensor in sorted(
            re.findall(r'graph->' + kind + r'\.tensors\[(\d+)\] = norm_tensor\[(\d+)\]', code),
            key=lambda pair: int(pair[0]))]
    inputs, outputs = port_order('input'), port_order('output')
    expected_in = ['b0_hidden'] + [f'b{b}_{name}' for name in ('past_k', 'past_v', 'mask', 'rope_cos', 'rope_sin') for b in range(width)]
    expected_out = [f'b{width-1}_attach_hidden_out/out0'] + [f'b{b}_attach_new_{name}/out0' for name in ('k', 'v') for b in range(width)]
    assert [graph[v]['name'] for v in inputs] == expected_in, inputs
    assert [graph[v]['name'] for v in outputs] == expected_out, outputs
    return {'input_order': inputs, 'output_order': outputs, 'contract_verified_from_exported_c': True}
