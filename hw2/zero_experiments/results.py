"""Read-only analysis of saved measurements; never creates training results."""
import csv
import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import t
from .runtime import write_csv, write_json

REFERENCE_STD = {'zero':.08,'kaiming':.13,'xavier':.16}
REFERENCE = {'zero':5.13,'kaiming':5.15,'xavier':5.23,'rezero':5.20,'fixup':5.17,'skipinit':5.26,'constnet':72.39,'constnet_star':5.41}

def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))

def interval(values):
    a = np.asarray(values,dtype=float)
    if len(a)==0:
        return {'n':0,'mean':None,'std':None,'ci_low':None,'ci_high':None}
    mean = float(a.mean())
    if len(a)<2:
        return {'n':len(a),'mean':mean,'std':None,'ci_low':None,'ci_high':None}
    std = float(a.std(ddof=1))
    half = float(t.ppf(.975,len(a)-1)*std/np.sqrt(len(a)))
    return {'n':len(a),'mean':mean,'std':std,'ci_low':mean-half,'ci_high':mean+half}

def aggregate(root):
    root = Path(root)
    records, configs = [], []
    for p in sorted(root.glob('resnet18-*/summary.json')):
        r = json.loads(p.read_text())
        c = json.loads((p.parent/'config.json').read_text())
        if r['status'] != 'complete':
            continue
        records.append(r)
        configs.append(c)
    # Each table must have one schedule/data/precision protocol. Runtime ConstNet mode may differ intentionally.
    keys = ('epochs','batch_size','lr','momentum','weight_decay','warmup_epochs','warmup_start_lr',
            'scheduler','train_limit','test_limit','precision','protocol','determinism')
    if configs and any(any(c[k]!=configs[0][k] for k in keys) for c in configs):
        raise ValueError('Incompatible training protocols cannot be aggregated')
    grouped = defaultdict(list)
    for r in records:
        grouped[r['method']].append(r)
    rows = []
    for method,runs in sorted(grouped.items()):
        if len({r['seed'] for r in runs})!=len(runs):
            raise ValueError('Duplicate seeds')
        errors = [r['test_error'] for r in runs]
        stats = interval(errors)
        rows.append({'method':method, **stats, 'std_ddof':1, 'ci_method':'Student-t 95%',
                     'seeds':';'.join(str(r['seed']) for r in runs), 'smoke':any(r['smoke'] for r in runs),
                     'paper_reference_error':REFERENCE.get(method), 'paper_reference_std':REFERENCE_STD.get(method),
                     'difference_pp':stats['mean']-REFERENCE[method] if method in REFERENCE else None})
    paired = []
    if 'zero' in grouped:
        z = {r['seed']:r['test_error'] for r in grouped['zero']}
        for method,runs in sorted(grouped.items()):
            if method=='zero':
                continue
            other = {r['seed']:r['test_error'] for r in runs}
            shared = sorted(z.keys() & other.keys())
            stats = interval([z[s]-other[s] for s in shared])
            zstd, ostd = interval(list(z.values()))['std'], interval(list(other.values()))['std']
            paired.append({'comparison':f'zero-minus-{method}',**stats,'matched_seeds':';'.join(map(str,shared)),
                           'seed_sets_equal':z.keys()==other.keys(),
                           'std_ratio_zero_over_other':zstd/ostd if zstd is not None and ostd not in (None,0) else None})
    write_csv(root/'aggregate.csv',rows)
    write_csv(root/'paired.csv',paired)
    write_json(root/'aggregate.json',{'runs':records,'aggregate':rows,'paired':paired})
    return rows, paired, records


def report(root, gradient_root=None):
    root = Path(root)
    rows, paired, runs = aggregate(root)
    figures = [p.name for p in (root/'plots').glob('*.png')] if (root/'plots').exists() else []
    grad = read_csv(Path(gradient_root)/'gradients.csv') if gradient_root else []
    groups = defaultdict(list)
    for r in grad:
        groups[(r['method'],r['mode'],r['seed'])].append(r)
    compression = list((root/'compression').glob('*/metadata.json'))
    full_done = {(r['method'],r['seed']) for r in runs if not r['smoke'] and r['epochs']==200
                 and r['train_images']==50000 and r['test_images']==10000}
    missing_main = [(m,seed) for m in ('zero','kaiming','xavier') for seed in range(10) if (m,seed) not in full_done]
    complete_grad = sum(len(v)==400 for v in groups.values())
    lines = ['# Фактический отчёт ResNet-18 / CIFAR-10','',
             f'Протокол и источники: [protocol.md]({os.path.relpath(Path(__file__).resolve().parents[1]/"protocol.md",root)}). Все числа ниже взяты из сохранённых измерений.',
             'Реконструкция содержит неизвестные авторские настройки; совпадение accuracy не подтверждает точное воспроизведение.','',
             '| Эксперимент | Фактическое покрытие |','|---|---|',
             f'| Table 2 | {30-len(missing_main)}/30 полных запусков; дополнительно {sum(r["smoke"] for r in runs)} smoke |',
             f'| Table 3 | {len([r for r in runs if r["method"] not in ("zero","kaiming","xavier")])} запусков альтернатив |',
             f'| Figure 5 | {len(list(root.glob("resnet18-*/ranks.csv")))} траекторий, включая smoke |',
             f'| Figure 6 | {len(compression)} обработанных checkpoint, статус в metadata.json |',
             f'| Figure 8 | {len(groups)} групп; {len(grad)} шагов; {complete_grad} групп с 400 шагами |','',
             'Std — sample ddof=1; CI — 95% Student-t; при n=1 std и CI не определены. Smoke не является результатом Table 2.','',
             '| Метод | n | Mean test error | Sample std | 95% CI | Ориентир статьи | Smoke |',
             '|---|---:|---:|---:|---|---:|---|']
    for r in rows:
        lines.append(f'| {r["method"]} | {r["n"]} | {r["mean"]:.4f} | {r["std"]} | [{r["ci_low"]}, {r["ci_high"]}] | {r["paper_reference_error"]} | {r["smoke"]} |')
    lines += ['', '## Отдельные запуски', '', '| Метод | Seed | Эпохи | Last error | Best observed test accuracy | Train seconds | Peak RSS MB |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in runs:
        lines.append(f'| {r["method"]} | {r["seed"]} | {r["epochs"]} | {r["test_error"]:.4f} | {r["best_observed_test"]["accuracy"]:.4f} | {r["train_seconds"]:.2f} | {r["process_peak_rss_mb"]:.1f} |')
    lines += ['', 'Best observed test — описательная верхняя оценка, не основной checkpoint и не tuning criterion.',
              'Ранги, test evaluation и checkpoint IO исключены из train_seconds; rank_seconds и test_seconds сохранены отдельно.',
              'RSS — high-water mark всего процесса, CUDA memory — max_memory_allocated; это разные измерения.','',
              '## Figure 8', '', '| Метод | Режим | Seed | Шагов | Первый LR |','|---|---|---:|---:|---:|']
    for (method,mode,seed), values in sorted(groups.items()):
        lines.append(f'| {method} | {mode} | {seed} | {len(values)} | {values[0]["lr"]} |')
    lines += ['', '## Графики', '']
    for name in figures:
        lines.append(f'![{name}](plots/{name})')
    lines += ['', '## Интерпретация и оставшиеся эксперименты', '',
              'Небольшое отличие средних не означает превосходства. Парные разности/CI и отношение std — в paired.csv; при разных наборах seed или smoke вывод о воспроизводимости ограничен.',
              'Stable rank не доказывает механизм GLRL. Kernel-rank панель использует выбранный stable rank; точный алгоритм Figure 6 не опубликован.',
              'Tucker сообщает параметры и MACs, не обещает ускорения. Полноранговые проверки записаны отдельно. Pruning не содержит fine-tuning.',
              f'Основная серия: недостающих полных запусков {len(missing_main)} из 30. Полная Figure 8 требует всех трёх методов в обоих режимах, 400 шагов каждой группы. Альтернативы и compression подтверждаются только записями соответствующих каталогов.',
              'Figure 4/depth, ResNet-50, ImageNet, MNIST и Transformer исключены по постановке.',
              'Возможные причины расхождений: неизвестные architecture/preprocessing/scheduler/weight-decay правила авторов, аппаратный nondeterminism, версия библиотек и короткое обучение.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    return root/'report.md'
