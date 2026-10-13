import unittest
from run_cpu_low_finalists import finalists


class FinalistSelectionTest(unittest.TestCase):
    def test_exact_input_length_formal_runs_and_all_objectives(self):
        def row(speed,ttft,total,repeats=5,aligned=True):
            return {'repeats':repeats,'aligned':aligned,'medians':{
                'decode_tokens_per_second':speed,'engine_ttft_seconds':ttft,'total_seconds':total}}
        entry={'runs':{'final-fast':row(20,1,8), 'n32-final-fast':row(25,1,6,aligned=False),
                       'n32-final-first':row(19,.3,7),'n32-final-total':row(24,.8,5),
                       'n128-final-fast':row(99,.01,1),'n32-final-unconfirmed':row(100,.01,1,2),
                       'n32-final-thermal':{'repeats':5,'status':'rejected-thermal-limit','medians':{}}}}
        self.assertEqual(set(finalists(entry,32)),{'n32-final-fast','n32-final-first','n32-final-total'})
        self.assertEqual(set(finalists(entry,64)),{'final-fast'})


if __name__=='__main__':unittest.main()
