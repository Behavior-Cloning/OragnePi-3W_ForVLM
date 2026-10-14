"""Regression for legal QK logits below the old finite mask sentinel."""
import unittest
import torch
from trained_probe import TrainedDecoder


class StrictMaskTest(unittest.TestCase):
    def test_large_negative_valid_score_cannot_attend_to_empty_slot(self):
        layer=object.__new__(TrainedDecoder)
        torch.nn.Module.__init__(layer)
        layer.heads=layer.kv=layer.hd=1
        layer.strict_mask=True
        layer.project=lambda *args:(torch.tensor([[[[-20000.]]]]),
                                   torch.ones(1,1,1,1),torch.full((1,1,1,1),7.))
        layer.finish=lambda hidden,attention:attention
        zeros=torch.zeros(1,1,2,1)
        mask=torch.tensor([[[[-10000.,-10000.,0.]]]])
        args=(torch.zeros(1,1,1),zeros,zeros,mask,None,None)
        self.assertEqual(layer(*args)[0].item(),7.)
        layer.strict_mask=False
        self.assertLess(layer(*args)[0].item(),.001)


if __name__=='__main__':unittest.main()
