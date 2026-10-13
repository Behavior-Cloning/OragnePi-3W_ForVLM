"""New MLP weight experiments must preserve attention and sensitive FP16."""
import copy
import unittest
import numpy as np
from fc_input_precision import convert_fc_weights_dfp


class WeightScopeTest(unittest.TestCase):
    def test_mse_scale_limit_is_enforced_for_outlier_matrix(self):
        graph={'Layers':{'gate':{'op':'fullconnect'}}}
        quant={'quantize_parameters':{'@gate:weight':{'qtype':'i8','quantizer':'perchannel_symmetric_affine'}}}
        matrix=np.full((4,65536),.13,np.float32);matrix[0,0]=32
        data={'gate':{'weight':matrix}}
        full=convert_fc_weights_dfp(graph,copy.deepcopy(quant),data,4,'gateup','mse')
        limited=convert_fc_weights_dfp(graph,copy.deepcopy(quant),data,4,'gateup','mse',1)
        peak=full['changes'][0]['scale_search'][0]['fl']
        self.assertEqual([r['fl'] for r in limited['changes'][0]['scale_search']],[peak,peak+1])
        self.assertLessEqual(limited['changes'][0]['after']['fl'],peak+1)
        self.assertGreater(full['changes'][0]['after']['fl'],peak+1)

    def test_mlp_preserves_attention_v_fix_and_sensitive_weights(self):
        sizes = {'q': (4, 4), 'k': (4, 2), 'v': (4, 2), 'o': (4, 4),
                 'gate': (4, 12), 'up': (4, 12), 'down': (12, 4), 'sensitive': (12, 4)}
        graph = {'Layers': {n: {'op': 'fullconnect'} for n in sizes}}
        quant = {'quantize_parameters': {'@' + n + ':weight':
                    {'qtype': 'i8', 'quantizer': 'perchannel_symmetric_affine'} for n in sizes}}
        qp = quant['quantize_parameters']
        qp['@v:weight'] = {'qtype': 'i8', 'quantizer': 'dynamic_fixed_point', 'fl': 5}
        qp['@sensitive:weight'] = {'qtype': 'f16', 'quantizer': 'float'}
        before = copy.deepcopy(qp)
        rng = np.random.default_rng(7)
        data = {n: {'weight': rng.normal(size=s).astype(np.float32)} for n, s in sizes.items()}
        report = convert_fc_weights_dfp(graph, quant, data, 4, 'mlp', 'mse')
        self.assertEqual({c['linear'] for c in report['changes']}, {'gate', 'up', 'down'})
        for name in ('q', 'k', 'v', 'o', 'sensitive'):
            self.assertEqual(qp['@' + name + ':weight'], before['@' + name + ':weight'])
        for change in report['changes']:
            selected = next(t for t in change['scale_search'] if t['fl'] == change['after']['fl'])
            self.assertLessEqual(selected['mse'], change['scale_search'][0]['mse'])


if __name__ == '__main__':
    unittest.main()
