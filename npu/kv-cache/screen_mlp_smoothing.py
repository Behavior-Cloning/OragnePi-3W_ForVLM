"""Host-only development screen for static A8 MLP channel rescaling.

This does not emulate the NPU compiler, claim board speed, or accept quality.
Calibration and checks must be declared development prompts, never holdouts.
All downstream arithmetic stays FP32 to isolate the gate/up quantization loss.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import gguf
import torch
import torch.nn.functional as F
from trained_probe import TrainedDecoder


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def restore_dfp(value, fl, bits):
    return (value * 2.**fl).round().clamp(-(2**(bits-1)), 2**(bits-1)-1) * 2.**(-fl)


def quantize_weight(value, max_extra_bits):
    peak = float(value.abs().max())
    assert math.isfinite(peak) and peak > 0
    peak_fl = math.floor(math.log2(127/peak))
    trials = []
    for fl in range(peak_fl, peak_fl+max_extra_bits+1):
        restored = restore_dfp(value, fl, 8)
        trials.append((float((restored-value).square().mean()), fl))
    _, fl = min(trials)
    return restore_dfp(value, fl, 8), fl


def errors(actual, expected):
    delta = (actual-expected).norm(dim=-1)
    per_token = delta/expected.norm(dim=-1).clamp_min(1e-20)
    return {'relative_l2': float((actual-expected).norm()/expected.norm().clamp_min(1e-20)),
            'per_token_l2_median': float(per_token.median()),
            'per_token_l2_p95': float(torch.quantile(per_token.flatten(), .95)),
            'per_token_l2_max': float(per_token.max())}


def mlp(layer, value, gate, up):
    return F.linear(F.silu(F.linear(value, gate))*F.linear(value, up), layer.down)


def collect(embedding, prompt, layers):
    ids = json.loads(prompt.read_text())['token_ids']
    assert ids and all(type(t) is int and 0 <= t < len(embedding) for t in ids)
    hidden = embedding[ids][None]
    samples = {}
    for index, layer in enumerate(layers):
        q,k,v = layer.project(hidden, *layer.rope(torch.arange(len(ids))))
        attention = F.scaled_dot_product_attention(q,
            k.repeat_interleave(layer.heads//layer.kv, 1),
            v.repeat_interleave(layer.heads//layer.kv, 1), is_causal=True)
        residual = hidden+F.linear(attention.transpose(1,2).reshape(1,-1,layer.dim), layer.o)
        activation = layer.norm(residual, layer.norm2)
        branch = mlp(layer, activation, layer.gate, layer.up)
        actual = residual+branch
        # Independently call the unmodified model path before collecting data.
        torch.testing.assert_close(actual, layer.full(hidden), rtol=2e-5, atol=1e-5)
        samples[index] = activation.clone()
        hidden = actual
    return samples


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--calibration', type=Path, nargs='+', required=True)
    p.add_argument('--check', type=Path, nargs='+', required=True)
    p.add_argument('--layers', type=int, default=2, choices=range(1,5))
    p.add_argument('--max-extra-bits', type=int, choices=(0,1), required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    assert not a.output.exists()
    assert all('holdout' not in str(path).lower() for path in a.calibration+a.check)
    assert not set(p.resolve() for p in a.calibration)&set(p.resolve() for p in a.check)
    torch.set_num_threads(2)
    reader=gguf.GGUFReader(a.model)
    t=next(t for t in reader.tensors if t.name=='token_embd.weight')
    embedding=torch.from_numpy(gguf.dequantize(t.data,t.tensor_type).copy())
    layers=[TrainedDecoder(reader,i,True).eval() for i in range(a.layers)]
    records=[]
    with torch.inference_mode():
        calibration=[collect(embedding,path,layers) for path in a.calibration]
        checks=[(path,collect(embedding,path,layers)) for path in a.check]
        for index,layer in enumerate(layers):
            values=torch.cat([v[index] for v in calibration],dim=1)
            amax=values.abs().amax(dim=(0,1)).clamp_min(1e-5)
            wmax=torch.maximum(layer.gate.abs().amax(dim=0),layer.up.abs().amax(dim=0)).clamp_min(1e-5)
            # Fixed small search. Alpha=0 is explicitly the unscaled control.
            for alpha in (0., .25, .5, .75):
                scale=(amax.pow(alpha)/wmax.pow(1-alpha) if alpha else torch.ones_like(amax))
                # Power-of-two factors make the FP16 folding auditable later.
                scale=2.**scale.log2().round().clamp(-8,8)
                scaled_values=values/scale
                fl=math.floor(math.log2(127/float(scaled_values.abs().max())))
                gate=layer.gate*scale;up=layer.up*scale
                qgate,gfl=quantize_weight(gate,a.max_extra_bits)
                qup,ufl=quantize_weight(up,a.max_extra_bits)
                for path,samples in checks:
                    value=samples[index];scaled=value/scale
                    original=mlp(layer,value,layer.gate,layer.up)
                    unquantized=mlp(layer,scaled,gate,up)
                    torch.testing.assert_close(unquantized,original,rtol=2e-4,atol=2e-5)
                    aq=restore_dfp(scaled,fl,8)
                    q8=mlp(layer,aq,qgate,qup)
                    weight_only=mlp(layer,scaled,qgate,qup)
                    integer=(scaled*2.**fl).round()
                    records.append({'layer':index,'alpha':alpha,'check':str(path),
                        'calibration_tokens':values.shape[1],'check_tokens':value.shape[1],
                        'activation_fl':fl,'gate_fl':gfl,'up_fl':ufl,
                        'scale_min':float(scale.min()),'scale_max':float(scale.max()),
                        'activation_clipped_fraction':float(((integer < -128)|(integer > 127)).float().mean()),
                        'fp32_identity':errors(unquantized,original),
                        'weight_only_mlp_error':errors(weight_only,original),
                        'w8a8_mlp_error':errors(q8,original)})
    output={'scope':__doc__, 'model':str(a.model),'model_sha256':digest(a.model),
            'layers':a.layers,'weight_dfp_max_extra_bits':a.max_extra_bits,
            'calibration':[{'path':str(path),'sha256':digest(path)} for path in a.calibration],
            'development_checks':[{'path':str(path),'sha256':digest(path)} for path in a.check],
            'records':records,'npu_execution_tested':False,'speed_measured':False,
            'independent_quality_accepted':False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(output,indent=2)+'\n')
    for index in range(a.layers):
        for alpha in (0., .25, .5, .75):
            r=[r for r in records if r['layer']==index and r['alpha']==alpha]
            print(index,alpha,'worst check p95 MLP relative error',max(v['w8a8_mlp_error']['per_token_l2_p95'] for v in r))


if __name__=='__main__':main()
