import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from zero_experiments.config import Config
from zero_experiments.models import create_model, METHODS
from zero_experiments.initialization import zero_init_
from zero_experiments.validation import check_initializations
from zero_experiments.training import train_one, learning_rate_at
from zero_experiments.ranks import stable_rank
from zero_experiments.data import make_loaders, set_epoch, batch_hash
from zero_experiments.results import interval

class ZerOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_initialization_all_parameters(self):
        self.assertTrue(check_initializations()['zero_seed_invariance'])
        for shape in ((3,3),(3,5),(7,9)):
            w = torch.empty(shape); zero_init_(w)
            self.assertTrue(torch.equal(w,torch.eye(*shape)))
        with self.assertRaises(ValueError):
            zero_init_(torch.empty(3,3,2,2))

    def test_ranks(self):
        self.assertEqual(stable_rank(torch.zeros(4,4)),0)
        self.assertEqual(stable_rank(torch.eye(4)),4)
        self.assertAlmostEqual(stable_rank(torch.diag(torch.tensor([3.,1.]))),10/9)
        self.assertAlmostEqual(stable_rank(torch.ones(3,8)),1)

    def test_schedule_statistics(self):
        self.assertEqual(learning_rate_at(0,1000,100,.1),.01)
        self.assertAlmostEqual(learning_rate_at(99,1000,100,.1),.1)
        self.assertEqual(learning_rate_at(0,1000,0,.1),.1)
        self.assertAlmostEqual(learning_rate_at(999,1000,100,.1),0)
        self.assertEqual(interval([1.,2.,3.])['std'],1.)
        self.assertIsNone(interval([1.])['std'])

    def test_explicit_alternative_blocks_finite_updates(self):
        for method in METHODS:
            model = create_model(method,0)
            if method in ('rezero','fixup','skipinit','zero_no_bn'):
                self.assertFalse(any(isinstance(m,nn.BatchNorm2d) for m in model.modules()))
            block = model.layer1[0]
            if method in ('rezero','skipinit'):
                self.assertEqual(block.alpha.item(),0)
            if method=='skipinit':
                self.assertTrue(hasattr(block,'merge_bias'))
            if method=='fixup':
                self.assertEqual(block.scale.item(),1)
                self.assertEqual(model.fc.weight.count_nonzero(),0)
            opt = torch.optim.SGD(model.parameters(),lr=.01)
            g = torch.Generator().manual_seed(2)
            x,y = torch.randn(4,3,32,32,generator=g),torch.tensor([0,1,2,3])
            for _ in range(3):
                opt.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(x),y)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                opt.step()
            if method=='zero':
                self.assertGreater(model.layer1[0].conv2.weight.count_nonzero(),0)
            del model,opt

    def test_resume_full_states(self):
        cfg = Config(methods=('zero',),seeds=(0,),epochs=2,warmup_epochs=1,batch_size=4,
                     save_rank_weights=False,cpu_threads=2,grad_steps=3)
        g = torch.Generator().manual_seed(19)
        ds = TensorDataset(torch.randn(8,3,32,32,generator=g),torch.arange(8))
        def loaders():
            return DataLoader(ds,batch_size=4,shuffle=True,generator=torch.Generator()),DataLoader(ds,batch_size=4)
        with tempfile.TemporaryDirectory() as tmp:
            a,b = Path(tmp)/'whole',Path(tmp)/'resume'
            train_one(cfg,'zero',0,a,loaders=loaders())
            train_one(cfg,'zero',0,b,loaders=loaders(),stop_after=1)
            train_one(cfg,'zero',0,b,resume=True,loaders=loaders())
            ca,cb = torch.load(a/'last.pt',weights_only=True),torch.load(b/'last.pt',weights_only=True)
            self.assertTrue(all(torch.equal(v,cb['model_state'][k]) for k,v in ca['model_state'].items()))
            self.assertEqual(ca['scheduler_state'],cb['scheduler_state'])
            self.assertTrue(torch.equal(ca['rng_state']['torch'],cb['rng_state']['torch']))
            self.assertTrue(all(torch.equal(v['momentum_buffer'],cb['optimizer_state']['state'][k]['momentum_buffer'])
                                for k,v in ca['optimizer_state']['state'].items()))
            self.assertEqual(json.loads((a/'gradients.json').read_text()),json.loads((b/'gradients.json').read_text()))
            with self.assertRaises(ValueError):
                train_one(replace(cfg,lr=.2),'zero',0,b,resume=True,loaders=loaders())

    def test_real_cifar_rng_isolation(self):
        if not Path('data/cifar-10-batches-py').exists():
            self.skipTest('Local CIFAR unavailable')
        cfg = Config(batch_size=4,train_limit=8,test_limit=4)
        loader,_ = make_loaders(cfg,7)
        set_epoch(loader,7,0)
        first = [batch_hash(x,y) for x,y in loader]
        create_model('xavier',581); torch.rand(100)
        set_epoch(loader,7,0)
        self.assertEqual(first,[batch_hash(x,y) for x,y in loader])
        worker_cfg = replace(cfg,num_workers=2)
        workers,_ = make_loaders(worker_cfg,7)
        set_epoch(workers,7,0)
        self.assertEqual(first,[batch_hash(x,y) for x,y in workers])
        set_epoch(loader,7,1)
        self.assertNotEqual(first,[batch_hash(x,y) for x,y in loader])

if __name__=='__main__':
    unittest.main()
