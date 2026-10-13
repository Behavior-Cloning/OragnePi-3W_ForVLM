"""A Query range fix must not touch weights, normalization, K/V or RoPE inputs."""
import copy
import unittest

from activation_ranges import widen_query_path


class QueryRangeTest(unittest.TestCase):
    def fixture(self):
        nodes={}
        def node(name,op,inputs=()):
            nodes[name]={'name':name,'op':op,'inputs':['@'+i+':out0' for i in inputs]}
        for name in ('hidden','rope_cos','rope_sin','past_k','past_v'):
            node(name,'input')
        node('norm','multiply',['hidden'])
        node('q','fullconnect',['norm'])
        node('k','fullconnect',['norm'])
        node('qview','reshape',['q'])
        node('qhalf','slice',['qview'])
        node('qneg','neg',['qhalf'])
        node('qrot','concat',['qhalf','qneg'])
        node('qcos','multiply',['qview','rope_cos'])
        node('qsin','multiply',['qrot','rope_sin'])
        node('qrope','add',['qcos','qsin'])
        node('qk','matmul',['qrope','past_k'])
        node('prob','softmax',['qk'])
        node('av','matmul',['prob','past_v'])
        q={'@'+k+':out0':{'qtype':'i16','quantizer':'dynamic_fixed_point','fl':11} for k in nodes}
        q['@q:weight']={'qtype':'i8','quantizer':'dynamic_fixed_point','fl':8}
        q['@q:bias']={'qtype':'i32','quantizer':'dynamic_fixed_point','fl':18}
        return {'Layers':nodes},{'quantize_parameters':q}

    def test_all_query_paths_and_only_query_paths_widen(self):
        graph,quant=self.fixture();before=copy.deepcopy(quant);original=copy.deepcopy(graph)
        report=widen_query_path(graph,quant,1)
        changed={k for k,v in quant['quantize_parameters'].items() if v!=before['quantize_parameters'][k]}
        expected={'@'+k+':out0' for k in ('q','qview','qhalf','qneg','qrot','qcos','qsin','qrope')}
        self.assertEqual(changed,expected)
        self.assertEqual(graph,original)
        self.assertEqual(report['query_projection'],'q')
        self.assertTrue(all(quant['quantize_parameters'][k]['fl']==10 for k in changed))

    def test_rejects_unexpected_query_dependency(self):
        graph,quant=self.fixture()
        graph['Layers']['qsin']['inputs']=['@past_v:out0']
        with self.assertRaises(AssertionError):
            widen_query_path(graph,quant,1)


if __name__=='__main__':unittest.main()
