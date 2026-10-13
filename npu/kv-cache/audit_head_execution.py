"""Separate decoder-propagated logits error from local output-head execution.

Use the board's actual final hidden states, not a synthetic two-layer input.
The decomposition is diagnostic FP32 arithmetic, not a proposed CPU offload.
"""
import argparse
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np
import torch
from exported_int8 import apply_exported_head
from trained_probe import LMHead


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--dtype', required=True)
    p.add_argument('--case', required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    folder = a.source/a.dtype/('board-'+a.case)
    out = folder/'head-execution-audit.json'
    assert not out.exists(), 'preserve previous audit'
    manifest = json.loads((a.source/'trained-manifest.json').read_text())
    runtime = json.loads((folder/'tokens.json').read_text())
    assert runtime['decoder_layers'] == manifest['language_layers_total']
    depth, fusion, dim = runtime['decoder_layers'], runtime['fusion_width'], manifest['dim']
    steps, vocab = runtime['generated_tokens'], manifest['vocab_size']
    trace = np.fromfile(folder/'decode-block-trace.f32', np.float32).reshape(steps-1, depth//fusion, dim)
    observed = np.fromfile(folder/'logits.f32', np.float32).reshape(steps, vocab)[1:]
    reference = np.load(folder/'fp32-reference-logits-w8.npy', mmap_mode='r')[1:]
    assert observed.shape == reference.shape
    reader = gguf.GGUFReader(a.model)
    arch = reader.fields['general.architecture'].contents()
    eps = reader.fields[arch+'.attention.layer_norm_rms_epsilon'].contents()
    head = LMHead(reader, eps).eval()
    with torch.inference_mode():
        constants = apply_exported_head(head, a.source, a.dtype, manifest)
        expected_on_actual_hidden = head(torch.from_numpy(trace[:, -1].copy())).numpy()
    assert expected_on_actual_hidden.shape == observed.shape
    assert all(np.isfinite(x).all() for x in (trace, observed, reference, expected_on_actual_hidden))
    local = observed.astype(np.float64)-expected_on_actual_hidden
    propagated = expected_on_actual_hidden.astype(np.float64)-reference
    total = observed.astype(np.float64)-reference
    norm2 = float(np.square(reference.astype(np.float64)).sum())
    local2 = float(np.square(local).sum())/norm2
    propagated2 = float(np.square(propagated).sum())/norm2
    total2 = float(np.square(total).sum())/norm2
    cross = 2*float((local*propagated).sum())/norm2
    assert np.isclose(local2+propagated2+cross, total2, rtol=1e-10, atol=1e-14)
    # Step 0 is prefill's first prediction and has no scalar block trace.
    metrics = {'all_error_relative_l2': total2**.5,
               'local_head_error_relative_l2': local2**.5,
               'decoder_propagated_error_relative_l2': propagated2**.5,
               'normalized_cross_term': cross,
               'local_head_argmax_agreement': float(np.mean(observed.argmax(-1)==expected_on_actual_hidden.argmax(-1))),
               'step_local_head_relative_l2': (np.linalg.norm(local,axis=-1)/np.linalg.norm(reference,axis=-1)).tolist(),
               'step_decoder_propagated_relative_l2': (np.linalg.norm(propagated,axis=-1)/np.linalg.norm(reference,axis=-1)).tolist()}
    expected_logp = torch.log_softmax(torch.from_numpy(expected_on_actual_hidden), dim=-1)
    actual_logp = torch.log_softmax(torch.from_numpy(observed), dim=-1)
    metrics['local_head_mean_kl_nats'] = float((expected_logp.exp()*(expected_logp-actual_logp)).sum(-1).mean())
    report = {'scope': __doc__, 'first_prediction_index': 1, 'prediction_count': steps-1,
              'decoder_layers': depth, 'vocab': vocab, 'metrics': metrics,
              'normalization': 'all L2 terms use the same full exported-reference logits norm; squared terms plus cross term equal total squared error',
              'source_sha256': {name: sha(folder/name) for name in
                                ('tokens.json','decode-block-trace.f32','logits.f32','fp32-reference-logits-w8.npy')},
              'head_constants': constants}
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in metrics.items() if not k.startswith('step_')}, indent=2))


if __name__ == '__main__':
    main()
