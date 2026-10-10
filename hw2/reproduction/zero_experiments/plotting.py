"""Eight panels from CSV measurements. Missing experiments are labeled, never synthesized."""
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .results import read_csv

COLORS = {'zero':'#0072BD','kaiming':'#00A000','xavier':'#FF0000'}
LABELS = {'zero':'ZerO','kaiming':'Kaiming','xavier':'Xavier'}

def curve(ax, rows, x, y, method, percent_x=False):
    selected = [r for r in rows if r['method']==method and r.get(y) not in (None,'')]
    by_seed = defaultdict(dict)
    for r in selected:
        by_seed[str(r['seed'])][float(r[x])] = float(r[y])
    if not by_seed:
        return
    # No interpolation/extrapolation; same seeds at every displayed point.
    xs = sorted(set.intersection(*(set(v) for v in by_seed.values())))
    if not xs:
        return
    a = np.array([[v[k] for k in xs] for v in by_seed.values()])
    mean = a.mean(0)
    ax.plot(xs,mean,label=LABELS.get(method,method),color=COLORS.get(method),marker='o' if len(xs)<20 else None,markersize=3)
    if len(a)>1:
        std = a.std(0,ddof=1)
        ax.fill_between(xs,mean-std,mean+std,color=COLORS.get(method),alpha=.13)


def finish(fig, axes, output, stem):
    for ax in np.asarray(axes).flat:
        ax.grid(alpha=.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
            missing = [LABELS[m] for m in COLORS if LABELS[m] not in labels]
            if missing:
                ax.text(.03,.03,'Missing: '+', '.join(missing),transform=ax.transAxes,fontsize=8)
        else:
            ax.text(.5,.5,'No recorded measurements',ha='center',transform=ax.transAxes)
    fig.tight_layout()
    output.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):
        fig.savefig(output/f'{stem}.{ext}',dpi=180,bbox_inches='tight')
    return fig


def pruning_curve(ax, rows, method):
    # Align by the fixed requested grid; plot mean *actual* remaining nonzeros.
    by_seed = defaultdict(dict)
    for r in rows:
        if r['method']==method:
            by_seed[r['seed']][float(r['requested_keep'])] = (float(r['actual_remaining_percent']),float(r['accuracy']))
    if not by_seed:
        return
    grid = sorted(set.intersection(*(set(v) for v in by_seed.values())))
    values = np.array([[v[k] for k in grid] for v in by_seed.values()])
    xs, ys = values[:,:,0].mean(0), values[:,:,1].mean(0)
    ax.plot(xs,ys,label=LABELS[method],color=COLORS[method],marker='o',markersize=3)
    if len(values)>1:
        spread = values[:,:,1].std(0,ddof=1)
        ax.fill_between(xs,ys-spread,ys+spread,color=COLORS[method],alpha=.13)


def annotate_scope(fig, root, gradient_rows=None):
    import json
    config = Path(root)/'run_config.json'
    smoke = False
    if config.exists():
        cfg = json.loads(config.read_text())
        smoke = cfg.get('train_limit') is not None or cfg.get('test_limit') is not None or cfg.get('epochs',200)<200
    if gradient_rows is not None:
        counts = defaultdict(int)
        for r in gradient_rows:
            counts[(r['method'],r['mode'],r['seed'])] += 1
        smoke = not counts or any(n<400 for n in counts.values())
    if smoke:
        fig.suptitle('SMOKE CHECK — short runs, not paper results',fontsize=10)


def figure5(root):
    root = Path(root)
    rows = sum((read_csv(p) for p in sorted(root.glob('resnet18-*/ranks.csv'))),[])
    if not rows:
        raise FileNotFoundError(f'Figure 5: ranks.csv absent under {root}; imported images cannot restore raw trajectories.')
    fig,axes = plt.subplots(1,3,figsize=(12,3.4))
    for ax,layer in zip(axes,('layer2.0.conv1','layer3.0.conv1','layer4.0.conv1')):
        for method in COLORS:
            curve(ax,[r for r in rows if r['layer']==layer],'epoch','stable_rank',method)
        ax.set(xlabel='Epoch',ylabel='Stable rank',title=layer)
    annotate_scope(fig,root)
    return finish(fig,axes,root/'plots','figure5-resnet18')


def figure6(root):
    root = Path(root)
    output = root/'compression'
    kernels, pruning, tucker = [], [], []
    for directory in sorted(output.glob('*')):
        kernels += read_csv(directory/'kernel_ranks.csv')
        pruning += read_csv(directory/'pruning.csv')
        tucker += read_csv(directory/'tucker.csv')
    fig,axes = plt.subplots(1,3,figsize=(12,3.4))
    for method in COLORS:
        curve(axes[0],kernels,'index','stable_rank',method)
        pruning_curve(axes[1],pruning,method)
        curve(axes[2],tucker,'r_out','accuracy',method)
    axes[0].set(title='Trained convolution kernels',xlabel='Order of convolution',ylabel='Kernel rank (stable)')
    axes[1].set(title='Local magnitude pruning',xlabel='Weights remaining (%) — actual nonzeros',ylabel='Test accuracy (%)')
    axes[1].invert_xaxis()
    axes[2].set(title='Tucker-2 (r_out = r_in)',xlabel='Channel rank',ylabel='Test accuracy (%)')
    axes[2].invert_xaxis()
    annotate_scope(fig,root)
    return finish(fig,axes,root/'plots','figure6-resnet18')


def load_gradient_rows(gradient_root):
    import json
    directory = Path(gradient_root)
    csv_path, json_path = directory/'gradients.csv', directory/'gradients.json'
    if csv_path.is_file():
        rows = read_csv(csv_path)
    elif json_path.is_file():
        rows = json.loads(json_path.read_text())
    else:
        raise FileNotFoundError(f'Figure 8: no gradients.csv or gradients.json in {directory.resolve()}. Set GRADIENT_ROOT to the measurement directory.')
    required = {'method','seed','mode','step','gradient_norm'}
    if not rows:
        raise ValueError(f'Figure 8: gradient measurements are empty in {directory.resolve()}')
    if any(not required.issubset(row) for row in rows):
        raise ValueError(f'Figure 8: invalid gradient columns in {directory.resolve()}; required: {sorted(required)}')
    if not any(row['mode'] in ('warmup','no_warmup') and row['method'] in COLORS for row in rows):
        raise ValueError(f'Figure 8: no recognized method/mode measurements in {directory.resolve()}')
    return rows


def figure8(root, gradient_root):
    root = Path(root)
    rows = load_gradient_rows(gradient_root)
    fig,axes = plt.subplots(1,2,figsize=(10,3.4))
    for ax,mode,title in zip(axes,('warmup','no_warmup'),('Warmup: initial LR 0.01','No warmup: initial LR 0.1')):
        for method in COLORS:
            curve(ax,[r for r in rows if r['mode']==mode],'step','gradient_norm',method)
        ax.set(title=title,xlabel='Iteration (zero-based)',ylabel='Stem gradient L2 norm')
    annotate_scope(fig,root,rows)
    return finish(fig,axes,root/'plots','figure8-resnet18')


def training_curves(root):
    root = Path(root)
    rows = []
    for p in root.glob('resnet18-*/history.csv'):
        import json
        cfg = json.loads((p.parent/'config.json').read_text())
        rows += [{'method':cfg['method'],'seed':cfg['seed'],**r} for r in read_csv(p)]
    if not rows:
        raise FileNotFoundError(f'Training curves: history.csv absent under {root}; preserve the imported plots.')
    fig,axes = plt.subplots(1,3,figsize=(12,3.4))
    for ax,metric in zip(axes,('train_loss','test_accuracy','lr')):
        for method in COLORS:
            curve(ax,rows,'epoch',metric,method)
        ax.set(xlabel='Epoch',ylabel=metric)
    annotate_scope(fig,root)
    return finish(fig,axes,root/'plots','training')


def plot_all(root, gradient_root=None):
    figs = [figure5(root),figure6(root),training_curves(root)]
    if gradient_root is not None:
        figs.append(figure8(root,gradient_root))
    return figs
