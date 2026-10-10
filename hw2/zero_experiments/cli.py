import argparse
from dataclasses import replace
from pathlib import Path
from .config import Config
from .models import METHODS, inspect_model
from .runtime import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description='ResNet-18 CIFAR-10 reconstruction; training/measurement only')
    parser.add_argument('command',choices=('inspect','check-init','smoke','train','seeds','alternatives','gradients','ranks','pruning','tucker','compression','benchmark','report'))
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/main.json')
    parser.add_argument('--root',type=Path,default=Path('runs/reproduction'))
    parser.add_argument('--gradient-root',type=Path)
    parser.add_argument('--device')
    parser.add_argument('--data-dir')
    parser.add_argument('--num-workers',type=int)
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--methods',nargs='+',choices=METHODS)
    parser.add_argument('--seeds',nargs='+',type=int)
    parser.add_argument('--determinism',choices=('strict','original'))
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--steps',type=int)
    parser.add_argument('--test-limit',type=int)
    parser.add_argument('--layer',default='layer4.1.conv2')
    parser.add_argument('--keep',nargs='+',type=float,default=[.9,.85,.8,.75,.7,.65,.6,.55,.5])
    parser.add_argument('--ranks',nargs='+',type=int,default=[60,50,40,30,25,20,15,10,5])
    args = parser.parse_args(argv)
    cfg = Config.load(args.config)
    overrides = {k:getattr(args,k) for k in ('device','data_dir','num_workers','epochs','methods','seeds','determinism','test_limit') if getattr(args,k) is not None}
    if args.epochs is not None:
        overrides['warmup_epochs'] = min(cfg.warmup_epochs,args.epochs)
    cfg = replace(cfg,**overrides)
    cfg.methods,cfg.seeds = tuple(cfg.methods),tuple(cfg.seeds)
    if args.command=='smoke':
        cfg = replace(cfg,epochs=2,warmup_epochs=1,batch_size=16,seeds=(0,),train_limit=128,test_limit=64)
    if args.command=='train' and args.seeds is None:
        cfg = replace(cfg,seeds=(0,))
    if args.command=='alternatives' and args.methods is None:
        cfg = replace(cfg,methods=('zero','rezero','fixup','skipinit','constnet','constnet_star'))
    cfg.validate()
    if args.command=='inspect':
        args.root.mkdir(parents=True,exist_ok=True)
        for method in cfg.methods:
            info = inspect_model(method)
            write_json(args.root/f'model-{method}.json',info)
            print(method,info['parameters'])
        return
    if args.command=='check-init':
        from .validation import check_initializations
        results = check_initializations()
        write_json(args.root/'initialization_checks.json',results)
        print(results)
        return
    if args.command in ('train','seeds','alternatives','smoke'):
        from .training import run_suite
        run_suite(cfg,args.root,args.resume)
    elif args.command=='benchmark':
        from .training import benchmark
        benchmark(cfg,args.root)
    elif args.command=='gradients':
        from .training import gradient_suite
        gradient_suite(cfg,args.root,args.steps)
    elif args.command in ('ranks','pruning','tucker','compression'):
        from .compression import analyze_checkpoint
        kind = 'all' if args.command=='compression' else args.command
        checkpoints = sorted(args.root.glob('resnet18-*/last.pt'))
        if not checkpoints:
            parser.error('No last.pt checkpoints under --root')
        for cp in checkpoints:
            if not any(cp.parent.name.startswith(f'resnet18-{m}-seed') for m in cfg.methods):
                continue
            output = args.root/'compression'/cp.parent.name
            analyze_checkpoint(cp.parent,cfg,output,kind,args.layer,args.keep,args.ranks)
    elif args.command=='report':
        from .results import report
        print(report(args.root,args.gradient_root))

if __name__=='__main__':
    main()
