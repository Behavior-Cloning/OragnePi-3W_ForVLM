"""Audit actual quantizers and reproduce the v1 headroom-override experiment.

The first implementation applied the MLP override to the global INT16 loop.
Its recorded CLI arguments therefore do not describe the effective ranges.
Version 2 fixes that routing; explicit swapped ranges reproduce those measured
candidates. Inspect artifacts rather than trusting directory names or options.
"""
import hashlib
import json
from pathlib import Path
import yaml

root = Path(__file__).resolve().parents[1] / 'results'
records = []
for source, baseline, candidate, layers, layer_bits, mlp_bits in (
    ('full-smol500-stable-v3-c128', 'w8-head-wide', 'w8-mlp-gateup-h1', [3,31], 1, 4),
    ('full-intern1b-sharded-v2-c128', 'w8-balanced-headwide', 'w8-mlp-gateup-h0', [2,3], 0, 1),
):
    for layer in layers:
        folder = root / source / candidate / f'l{layer}'
        prior = root / source / baseline / f'l{layer}'
        reproduction = root / source / 'w8-mlp-reproduced' / f'l{layer}'
        conversion = json.loads((folder / 'conversion.json').read_text())
        calibration = root / conversion['base']
        def quant(path):
            return yaml.safe_load((path / 'decoder.quantize').read_text())['quantize_parameters']
        old, actual, calibrated = quant(prior), quant(folder), quant(calibration)
        assert actual == quant(reproduction), 'reproduced quantization semantics differ'
        graph=json.loads((folder/'decoder.json').read_text())['Layers']
        mlp_nodes = {n for v in conversion['int8_mlp_projections'].values() for n in v['activation_nodes']}
        allowed_mlp = {'@'+n+':out0' for n in mlp_nodes}
        for v in conversion['int8_mlp_projections'].values():
            allowed_mlp.update(('@'+v['linear']+':weight','@'+v['linear']+':bias'))
        deltas = []
        for key in sorted(old.keys() | actual.keys()):
            if old.get(key) == actual.get(key):
                continue
            before, after = old.get(key), actual.get(key)
            if key not in allowed_mlp:
                assert key.endswith(':out0') and before['qtype'] == after['qtype'] == 'i16'
                assert {k:v for k,v in before.items() if k!='fl'} == {k:v for k,v in after.items() if k!='fl'}
            deltas.append({'tensor':key,'old':before,'new':after,'mlp_boundary':key in allowed_mlp})
        for key, value in actual.items():
            if key.endswith(':out0') and value['qtype']=='i16':
                node=key[1:].split(':')[0]
                if graph[node]['op']=='variable':
                    continue
                expected=mlp_bits if node in mlp_nodes else layer_bits
                assert calibrated[key]['fl']-value['fl']==expected,(key,expected)
        hashes={}
        for filename in ('decoder.json','decoder.data','decoder.quantize','tensor-inputmeta.yml','export_nbg_unify/network_binary.nb'):
            def digest(path):
                with path.open('rb') as stream:
                    return hashlib.file_digest(stream,'sha256').hexdigest()
            original_digest,new_digest=digest(folder/filename),digest(reproduction/filename)
            hashes[filename]={'measured':original_digest,'reproduced':new_digest,'identical':original_digest==new_digest}
            if filename!='decoder.quantize':
                assert original_digest==new_digest,(source,layer,filename)
        records.append({'source':source,'layer':layer,'measured_variant':candidate,
            'effective_mlp_headroom_bits':mlp_bits,'effective_other_headroom_bits_in_this_layer':layer_bits,
            'quantization_differences':deltas,'reproduction_sha256':hashes,
            'reproduction_conversion':json.loads((reproduction/'conversion.json').read_text())})
report={'scope':'actual per-tensor quantizers; corrected compiler reproduces quantization semantics and NBG bytes. YAML key order may differ.',
        'erratum':'v1 MLP headroom option was routed to non-MLP INT16 tensors. Measured candidate narrows attention/KV ranges only in the two sensitive layers; MLP keeps original headroom. v2 fixes option routing.',
        'layers':records}
out=root/'mlp-precision-audit.json'
out.write_text(json.dumps(report,indent=2)+'\n')
print(out)
print('Four layers: quantization semantics identical; NBG binaries reproduced byte for byte.')
