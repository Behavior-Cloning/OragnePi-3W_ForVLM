"""Map independently decoded compiler constants into the FP32 graph interpreter."""
import copy
import re

import gguf
import numpy as np
from exported_int8 import apply_exported_int8
from trained_probe import TrainedDecoder


def restore(graph, data, folder, conversion, model):
    reader = gguf.GGUFReader(model)
    result = copy.deepcopy(data)
    for i in range(conversion['fusion_width']):
        prefix = f'b{i}_'
        layer = TrainedDecoder(reader, conversion['first_layer']+i, True).eval()
        mapping = {}
        for node, parameters in data.items():
            if not node.startswith(prefix) or graph['Layers'][node]['op'] != 'fullconnect':
                continue
            original = parameters['weight']
            matches = [key for key in ('q', 'k', 'v', 'o', 'gate', 'up', 'down')
                       if getattr(layer, key).shape == original.T.shape and
                       np.array_equal(getattr(layer, key).numpy(), original.T)]
            assert len(matches) == 1, (node, matches)
            mapping[node] = matches[0]
        assert len(mapping) == 7
        apply_exported_int8(layer, folder, prefix=prefix)
        for node, key in mapping.items():
            result[node]['weight'] = getattr(layer, key).numpy().T.copy()
            bias = getattr(layer, key+'_bias', None)
            if bias is not None:
                result[node]['bias'] = bias.numpy().copy()
            else:
                assert not np.any(result[node]['bias'])
        for key in ('norm1', 'norm2', 'q_norm', 'k_norm'):
            target = getattr(layer, key, None)
            if target is None:
                continue
            matches = [n for n in data if re.fullmatch(re.escape(prefix+key)+r'_\d+', n)]
            assert len(matches) == 1, (prefix, key, matches)
            node = matches[0]
            result[node]['data'] = target.numpy().reshape(data[node]['data'].shape).copy()
    return result
