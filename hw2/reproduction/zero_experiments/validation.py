import torch
from .models import create_model, BasicBlock
from .initialization import hadamard_matrix, zero_init_


def check_initializations():
    torch.set_num_threads(2)
    results = {}
    for method in ('zero','kaiming','xavier'):
        a,b = create_model(method,0),create_model(method,123)
        equal = all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
        assert equal == (method=='zero')
        results[method+'_seed_invariance'] = equal
        if method=='zero':
            for layer in a.modules():
                if isinstance(layer,torch.nn.Conv2d):
                    outside = layer.weight.detach().clone()
                    outside[:,:,outside.shape[2]//2,outside.shape[3]//2] = 0
                    assert not torch.count_nonzero(outside)
                if isinstance(layer,BasicBlock):
                    assert not torch.count_nonzero(layer.conv2.weight)
                    assert torch.all(layer.bn2.weight==1)
            # Verify every trainable tensor incl stem, projections and FC against its rule.
            for name,p in a.named_parameters():
                if p.ndim in (2,4):
                    expected = torch.empty_like(p)
                    zero_init_(expected)
                    if name.endswith('conv2.weight'):
                        expected.zero_()
                    assert torch.equal(p,expected),name
                else:
                    assert torch.all(p == (1 if name.endswith('weight') else 0)),name
        del a,b
    h = hadamard_matrix(8)
    assert torch.equal(h@h.T,8*torch.eye(8))
    w = torch.empty(5,3,3,3)
    zero_init_(w)
    assert torch.equal(w[:,:,1,1],h[:5,:3]*.5)
    results['paper_hadamard_factor'] = 'sqrt(2/N)'
    results['central_kernel_and_all_parameters'] = 'passed'
    return results
