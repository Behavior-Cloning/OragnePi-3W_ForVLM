"""Export and verify adversarial activation inputs on the actual VIPLite backend."""
import argparse
import json
from pathlib import Path
import numpy as np
import onnx
import torch
from torch import nn


class Activation(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def forward(self, x):
        if self.kind == 'softmax':
            return torch.softmax(x, -1)
        if self.kind == 'softmax_explicit_max':
            return torch.softmax(x - x.amax(-1, keepdim=True), -1)
        if self.kind == 'silu':
            return torch.nn.functional.silu(x)
        divisor = x.abs().amax(-1, keepdim=True).clamp_min(1e-5**0.5 if self.kind in ('stable_rmsnorm_v2','stable_rmsnorm_v3') else 1.0)
        scaled = x / divisor
        epsilon = (1e-5**0.5 / divisor).square() if self.kind=='stable_rmsnorm_v3' else 1e-5 / divisor / divisor
        return scaled * torch.rsqrt((scaled * scaled).mean(-1, keepdim=True) + epsilon)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--validate', action='store_true')
    args = p.parse_args()
    folder = args.folder
    folder.mkdir(parents=True, exist_ok=True)
    if args.validate:
        manifest = json.loads((folder/'activation-manifest.json').read_text())
        records = []
        for case in manifest:
            actual = np.fromfile(folder/(case['name']+'-output.f32'), np.float32)
            expected = np.fromfile(folder/(case['name']+'-reference.f32'), np.float32)
            assert actual.shape == expected.shape
            finite = bool(np.isfinite(actual).all())
            record = {**case, 'finite': finite,
                      'max_abs_error': float(np.max(np.abs(actual-expected))),
                      'relative_l2': float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected), 1e-12))}
            if case['name'].startswith('softmax'):
                rows = actual.reshape(-1, 129)
                record['max_probability_sum_error'] = float(np.max(np.abs(rows.sum(-1)-1)))
                record['min_probability'] = float(rows.min())
            record['passed'] = finite and bool(np.allclose(actual, expected, atol=5e-3, rtol=1e-2))
            per_case=[]
            for a,b in zip(actual.reshape(case['cases'],-1),expected.reshape(case['cases'],-1)):
                per_case.append({'max_abs_error':float(np.max(np.abs(a-b))),
                                 'relative_l2':float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12))})
            record['per_case']=per_case
            record['passed'] = record['passed'] and max(r['relative_l2'] for r in per_case)<0.01
            records.append(record)
        (folder/'activation-validation.json').write_text(json.dumps(records, indent=2)+'\n')
        print(json.dumps(records, indent=2))
        if not all(r['passed'] for r in records if r['name'] not in ('stable_rmsnorm','stable_rmsnorm_v2')):
            raise SystemExit(3)
        return
    torch.set_num_threads(1)
    records = []
    for index, name in enumerate(('softmax', 'softmax_explicit_max', 'silu', 'stable_rmsnorm', 'stable_rmsnorm_v2','stable_rmsnorm_v3')):
        # Use identical inputs across equivalent formulas for paired comparisons.
        rng = np.random.default_rng(42 if name.startswith('softmax') else (43 if name=='silu' else 44))
        shape = (1, 15, 1, 129) if name.startswith('softmax') else (1, 1, 2560 if name=='silu' else 960)
        base = rng.normal(size=shape).astype(np.float32)
        if name.startswith('softmax'):
            # Each case has 15 attention rows; offsets exercise overflow resistance.
            cases = [base, base+1000, base-1000, base+10000, np.zeros(shape, np.float32),
                     np.full(shape, -10000, np.float32)]
            masked = np.full(shape, -10000, np.float32)
            masked[..., -1] = 0
            extreme = np.full(shape, -60000, np.float32)
            extreme[..., -1] = 60000
            cases += [masked, extreme]
        elif name=='silu':
            cases = [np.linspace(-60000, 60000, shape[-1], dtype=np.float32).reshape(shape),
                     np.linspace(-30, 30, shape[-1], dtype=np.float32).reshape(shape), base]
        else:
            cases = [base*s for s in (0, 1e-6, 1e-3, 1, 100, 10000)]
            outlier = base.copy();outlier[..., 0]=60000;outlier[..., 1]=-60000
            cases.append(outlier)
        # Compare the same representable inputs, separating IO rounding from kernel error.
        cases = [v.astype(np.float16).astype(np.float32) for v in cases]
        module = Activation(name).eval()
        path = folder/f'layer{index}.onnx'
        torch.onnx.export(module, torch.from_numpy(cases[0]), str(path), input_names=['hidden'],
                          output_names=['output'], opset_version=13, dynamo=False)
        model = onnx.load(path);model.ir_version=8;onnx.checker.check_model(model);onnx.save(model,path)
        references=[];entries=[]
        for j,value in enumerate(cases):
            with torch.inference_mode():
                if name.startswith('stable_rmsnorm'):
                    x=torch.from_numpy(value)
                    reference=x*torch.rsqrt((x*x).mean(-1,keepdim=True)+1e-5)
                else:
                    reference=module(torch.from_numpy(value))
            references.append(reference.numpy())
            filename=f'{name}-{j}.npy';np.save(folder/filename,value);entries.append(filename)
        np.asarray(cases).tofile(folder/(name+'-input.f32'))
        np.asarray(references).tofile(folder/(name+'-reference.f32'))
        (folder/f'layer{index}-dataset.txt').write_text('\n'.join(entries)+'\n')
        records.append({'name':name,'layer':index,'shape':shape,'count':int(np.prod(shape)),
                        'cases':len(cases),'input_rounding':'FP16 then FP32 oracle',
                        'tolerance':{'atol':5e-3,'rtol':1e-2}})
    (folder/'activation-manifest.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
