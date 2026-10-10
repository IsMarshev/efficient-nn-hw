# ZerO Initialization — ResNet-18 / CIFAR-10

Обучение и все измерения находятся в **Python-модулях** `zero_experiments/`. [`zero-init.ipynb`](zero-init.ipynb) только читает результаты, строит графики и анализирует их. Импорт модулей не начинает обучение; Python не исполняет ячейки notebook.

Реализованы восемь панелей: **Figure 5, верхний ряд (3), Figure 6 (3), Figure 8 (2)**. Локально выполнены smoke-проверки; полная серия 3×10×200 epochs ещё не обучена. CUDA недоступна. Подробности источников и всех предположений: [`protocol.md`](protocol.md). Итог локальной работы: [`REPORT.md`](REPORT.md).

В официальном репозитории нет CIFAR ResNet/training/compression code, поэтому точную архитектуру и весь протокол авторов восстановить нельзя. Это рабочая, явно документированная реконструкция. Статья v3 и официальный commit зафиксированы в `sources/manifest.json`; прежний notebook сохранён в `sources/legacy-zero-init.ipynb`.

## Установка и проверки

Рабочая папка — `hw2`. Установите PyTorch/torchvision для вашего CPU/CUDA, затем:

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_zero test_figure6
python run_zero.py check-init --root runs/checks
python run_zero.py inspect --root runs/checks
```

`inspect` сохраняет полную модель, формы всех операций и число параметров. Основная BN ResNet-18: **11 173 962 параметра**, CIFAR stem 3×3/stride1, без maxpool, BasicBlock [2,2,2,2], channels 64/128/256/512, adaptive pool, Linear512→10. Не используется ImageNet `torchvision.models.resnet18`.

## Локальная проверка всей цепочки

```bash
python run_zero.py smoke --device cpu --root runs/reproduction-smoke
python run_zero.py gradients --config configs/smoke.json --root runs/figure8-smoke --steps 8
python run_zero.py compression --config configs/smoke.json --root runs/reproduction-smoke --test-limit 64 --keep .9 .7 .5 --ranks 20 5
python run_zero.py report --root runs/reproduction-smoke --gradient-root runs/figure8-smoke
```

Эти команды уже выполнены в данном workspace. Для повторного smoke выберите новый `--root` либо используйте `--resume`. Smoke: три метода, seed0, две эпохи, 128 train /64 test images, batch16. Figure 8 smoke: 8 шагов каждой из шести групп. Compression smoke: исходные коротко обученные checkpoints, 64 test images, сокращённые сетки. Низкая accuracy smoke не характеризует качество методов.

Откройте `zero-init.ipynb` и выполните все ячейки. В первой ячейке задаются `ROOT` и `GRADIENT_ROOT`. Notebook сохраняет PNG/PDF в `<ROOT>/plots/`; CSV/JSON и отчёт — в `<ROOT>/`. Не выполняет обучение или сжатие.

## Полные эксперименты на CUDA

```bash
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
python run_zero.py benchmark --config configs/main.json --device cuda --root runs/gpu-benchmark

# Один метод / seed, полный датасет и 200 эпох:
python run_zero.py train --config configs/main.json --methods zero --seeds 0 --device cuda --num-workers 4 --root runs/single-zero

# Основная серия: ZerO, Kaiming, Xavier × seeds 0..9:
python run_zero.py seeds --config configs/main.json --device cuda --num-workers 4 --root runs/main

# Отдельные шесть запусков для Figure 8, ровно 400 шагов каждого:
python run_zero.py gradients --config configs/figure8.json --device cuda --num-workers 4 --root runs/figure8 --steps 400

# Все три панели Figure 6 из исходных last.pt, на полном test:
python run_zero.py compression --config configs/main.json --device cuda --num-workers 4 --root runs/main

# Альтернативы, включая ConstNet*, на nondeterministic CUDA:
python run_zero.py alternatives --config configs/alternatives.json --num-workers 4 --root runs/alternatives

python run_zero.py report --root runs/main --gradient-root runs/figure8
python run_zero.py report --root runs/alternatives
```

Для notebook полной серии задайте `ROOT = BASE / "runs/main"`, `GRADIENT_ROOT = BASE / "runs/figure8"` и выполните все ячейки: это построит все 8 панелей из ваших измерений. До этого отсутствующие результаты обозначаются, кривые не дорисовываются.

`main.json`: SGD LR .1, momentum .9, WD 1e-4, warmup 10 epochs от .01. Batch128, 200 epochs и cosine после линейного warmup — зафиксированные детали реконструкции. WD применяется ко всем trainable tensors. Инициализации Conv: Kaiming normal fan_out/ReLU; Xavier normal gain1. FP32, TF32=False, strict determinism.

Strict и original разделены `--determinism strict|original`. ConstNet всегда strict; ConstNet* требует original и CUDA, разрешает nondeterministic cuDNN, без искусственного шума/dropout. Современный cuDNN не гарантирует воспроизведение symmetry-breaking старой GPU-платформы. Альтернативные методы меняют блоки/нормализацию/скаляры, а не только init-function. Реконструкции описаны отдельно в протоколе. `zero_no_bn` — дополнительный режим.

## Сжатие и ранги отдельно

```bash
python run_zero.py ranks --device cuda --root runs/main
python run_zero.py pruning --device cuda --root runs/main --keep .9 .85 .8 .75 .7 .65 .6 .55 .5
python run_zero.py tucker --device cuda --root runs/main --layer layer4.1.conv2 --ranks 60 50 40 30 25 20 15 10 5
```

Stable rank: полное ядро `[Cout,Cin*kH*kW]`, identity не вычитается. Слои траекторий — `layer2.0.conv1`, `layer3.0.conv1`, `layer4.0.conv1`; нулевому ядру условно присваивается 0. Каждый epoch сохраняет эти ядра. Figure 6 слева: stem +16 residual Conv, без projection shortcuts. Отдельного алгоритма Kernel Rank у авторов нет; используем stable rank по §5. Это не numerical rank MNIST.

Pruning: local magnitude внутри Conv/Linear, включая projections, исключая BN/bias. Маски независимы, fine-tuning отсутствует. Записываются mask slots и фактические nonzeros; график показывает actual remaining percent.

Tucker-2: HOSVD+5 HOOI, `(r_out,r_in)=(r,r)`, 512-channel `layer4.1.conv2`. Реальная факторизованная операция `1×1 → 3×3 → 1×1`. Полные ранги проверяются на одинаковом входе; FP32 absolute tolerance `5e-6 * max(1,max|output|)`, relative tolerance `2e-4`; отдельно relative Frobenius error. Сохраняются параметры и Conv/Linear MACs на изображение 32×32. Latency не измеряется, ускорение не заявляется.

## Resume, файлы и статистика

```bash
python run_zero.py seeds --config configs/main.json --device cuda --num-workers 4 --root runs/main --resume
```

При resume сохраняйте исходные config/methods/seeds/num-workers. Допускается смена device, data_dir, cpu_threads; другая платформа не обещает битовую идентичность. `last.pt` содержит model, optimizer momentum, scheduler step, RNG Python/NumPy/Torch/CUDA/MPS, history/rank rows и metadata. Незаконченная эпоха пересчитывается от последней полной. Расписание/число эпох изменять нельзя.

Структура результатов:

```text
<ROOT>/run_config.json, summary.csv, aggregate.csv/json, paired.csv, report.md
<ROOT>/resnet18-<method>-seed<N>/
  environment.json, config.json, summary.json
  history.csv/json, ranks.csv, gradients.csv/json
  last.pt, best_observed_test.pt
  rank_weights/epoch-000.pt ... epoch-200.pt
<ROOT>/compression/resnet18-<method>-seed<N>/
  metadata.json, kernel_ranks.csv, pruning.csv, tucker.csv
  tucker_full_rank_check.json
<ROOT>/plots/*.png, *.pdf
<GRADIENT_ROOT>/gradients.csv/json, *-metadata.json
```

Данные, checkpoints и локальные runs не добавляются в git. Сохраняйте весь каталог результатов для переноса на другую машину. Для сырой истории выбранных ядер требуется около **1,24 GB на один 200-epoch run** /37,3 GB для 30 runs, плюс checkpoints. `benchmark` оценивает скорость, память и диск до полного обучения; модель замера не используется в экспериментах.

Начальные ZerO параметры одинаковы при разных seed. Random init, DataLoader shuffle и augmentation используют независимые RNG domains. Crop/flip детерминированы по run seed/epoch/original image index и не зависят от worker count. Figure 8 сравнивает SHA256 реально аугментированных batches между всеми шестью группами.

Last test error — основной результат. Best observed test отдельно, не гиперпараметрический tuning. Sample std ddof=1 и 95% Student-t CI, paired differences по одинаковым seeds, std ratio. При n=1 std/CI undefined. Конфигурации с разными протоколами не объединяются. В таблицах published reference хранится отдельно от measured error. Малое различие средних и низкий stable rank не доказывают превосходство или механизм GLRL.
