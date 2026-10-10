import unittest
import torch
from torch import nn
from zero_experiments.compression import magnitude_prune, tucker2_approx, channel_bases


class CompressionTests(unittest.TestCase):
    def test_tucker_full_rank_reconstruction_and_channel_ranks(self):
        torch.set_num_threads(2)
        torch.manual_seed(5)
        w = torch.randn(4, 4, 3, 3)
        original = w.clone()
        bases = channel_bases(w)
        full = tucker2_approx(w, 4, iterations=2, bases=bases)
        self.assertTrue(torch.allclose(full, w, atol=3e-6, rtol=3e-6))
        small = tucker2_approx(w, 2, iterations=3, bases=bases)
        for unfolded in (small.flatten(1), small.permute(1, 0, 2, 3).reshape(4, -1)):
            self.assertLessEqual(torch.linalg.matrix_rank(unfolded, atol=1e-4).item(), 2)
        initial = tucker2_approx(w, 2, iterations=0, bases=bases)
        self.assertLessEqual(torch.norm(w-small).item(), torch.norm(w-initial).item()+1e-5)
        self.assertTrue(torch.equal(w, original))

    def test_pruning_is_per_layer_and_non_cumulative(self):
        model = nn.Sequential(nn.Conv2d(2, 2, 1, bias=True), nn.BatchNorm2d(2))
        with torch.no_grad():
            model[0].weight.copy_(torch.tensor([1., -4., 2., -3.]).reshape(2, 2, 1, 1))
        state = {k: v.clone() for k, v in model.state_dict().items()}
        result = magnitude_prune(model, state, .5)
        self.assertEqual(result['retained_slots'], 2)
        self.assertTrue(torch.equal(model[0].weight.flatten(), torch.tensor([0., -4., 0., -3.])))
        self.assertTrue(torch.equal(model[0].bias, state['0.bias']))
        self.assertTrue(torch.equal(model[1].weight, state['1.weight']))
        magnitude_prune(model, state, 1.)
        self.assertTrue(torch.equal(model[0].weight, state['0.weight']))
        self.assertEqual(torch.count_nonzero(state['0.weight']).item(), 4)

class FactorizedConvolutionTests(unittest.TestCase):
    def test_full_rank_stride_padding_bias(self):
        from zero_experiments.compression import factorized_conv, full_rank_check
        for cin,cout,stride,bias in ((3,5,2,True),(4,4,1,False)):
            conv = nn.Conv2d(cin,cout,3,stride,padding=2,dilation=2,bias=bias)
            self.assertLess(full_rank_check(conv)['max_abs_error'],2e-5)
            fact = factorized_conv(conv,(cout,cin),iterations=0)
            self.assertEqual(fact[1].stride,conv.stride)
            self.assertEqual(fact[1].padding,conv.padding)
            self.assertEqual(fact[1].dilation,conv.dilation)
            if bias:
                self.assertTrue(torch.equal(fact[2].bias,conv.bias))

    def test_reduced_rectangular_ranks_match_dense_approximation(self):
        from zero_experiments.compression import factorized_conv, tucker_factors
        conv = nn.Conv2d(4,6,3,stride=2,padding=1,bias=True)
        u,core,v = tucker_factors(conv.weight,(3,2),iterations=2)
        dense = nn.Conv2d(4,6,3,stride=2,padding=1,bias=True)
        with torch.no_grad():
            dense.weight.copy_(torch.einsum('oa,abhw,ib->oihw',u,core,v))
            dense.bias.copy_(conv.bias)
        fact = factorized_conv(conv,(3,2),iterations=2)
        x = torch.randn(2,4,13,11)
        self.assertTrue(torch.allclose(fact(x),dense(x),atol=2e-6,rtol=2e-5))


if __name__ == "__main__":
    unittest.main()
