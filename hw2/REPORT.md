# Результат реализации в hw2

Обучение и измерения перенесены в `zero_experiments/`; `zero-init.ipynb` выполняет только анализ сохранённых CSV/JSON и построение графиков. Notebook полностью выполнен локально; все восемь требуемых панелей показаны на **реальных smoke-измерениях**, сохранены в PNG/PDF. Это проверка цепочки, не воспроизведение итоговых чисел статьи.

Источники и неизвестные настройки: [protocol.md](protocol.md). Фиксированы arXiv:2110.12661v3 и официальный commit `cf2bf5d7d967958b3bb232755b288d97239896a8`. В репозитории авторов нет CIFAR ResNet/training/compression. Различие Hadamard normalization между формулой статьи и MNIST-кодом описано явно; используется `sqrt(2/N)`.

| Эксперимент | Реально выполнено | Осталось |
|---|---|---|
| Table 2 | ZerO/Kaiming/Xavier, seed0, 2 epochs, 128 train /64 test | 30 полных запусков ×200 epochs, seeds0..9 |
| Table 3 | ReZero/Fixup/SkipInit/ConstNet по 2 smoke epochs; отдельно ZerO без BN | Полные альтернативы, ConstNet* на CUDA |
| Figure 5 верхние 3 панели | initial +epochs1/2, 3 метода; сырые ядра и CSV | Траектории полных 200 epochs |
| Figure 6, слева | 17 Conv, mapping и stable rank для трёх smoke checkpoints | Полностью обученные checkpoints |
| Figure 6, центр | local keep=.9/.7/.5, независимые точки, 64 test images | Полная сетка, 10k test, полные checkpoints |
| Figure 6, справа | r=20/5, настоящий factorized Conv, 64 test images | Полная сетка на обученных checkpoints |
| Figure 8, 2 панели | 3 methods×2 modes×8 steps=48 реальных измерений; одинаковые batch hashes | 400 steps каждой из6 групп на полном датасете |
| Correctness | 10 CPU tests +выполнение notebook | Проверка CUDA/ConstNet* на целевой машине |

Фактические smoke test accuracy последней эпохи (64 изображения):

| Метод | Accuracy, % | Train seconds, 2 эпохи |
|---|---:|---:|
| ZerO | 9.375 | 3.88 |
| Kaiming | 9.375 | 4.21 |
| Xavier | 10.9375 | 3.79 |
| ReZero | 4.6875 | 3.69 |
| Fixup | 4.6875 | 3.71 |
| SkipInit | 4.6875 | 3.78 |
| ConstNet | 14.0625 | 3.81 |
| ZerO без BN, дополнительно | 3.125 | 3.78 |

n=1 для каждого метода: std и confidence interval **не определены**. Полные ориентиры Table 2: error ZerO 5.13±.08, Kaiming5.15±.13, Xavier5.23±.16%. Эти опубликованные значения не являются локальными измерениями. По результатам короткого smoke нельзя сравнивать качество или воспроизводимость методов.

## Проверки и артефакты

Прошли проверки всех trainable tensors ZerO, seed invariance/variation, identity/partial identity/Hadamard, центральной инициализации Conv и нулевой conv2 при BN gamma=1. Loss/gradients конечны, residual conv2 обновляется после начала обучения. CPU resume совпадает по весам, momentum, scheduler state, RNG и gradient logs. Проверены RNG isolation от создания модели и инвариантность crop/flip к worker count. Проверены stable rank (zero/identity/rank1/diagonal), локальный независимый pruning, Tucker reconstruction/канальные ранги и executable factorized Conv с stride/padding/dilation/bias.

Полноранговый Tucker-2 на 512→512 smoke ядрах: relative Frobenius output errors **1.80e-6 /1.81e-6 /1.83e-6** для ZerO/Kaiming/Xavier. Maximum absolute errors и scale-aware tolerances сохранены в `tucker_full_rank_check.json`. Параметры и MACs каждого сжатого варианта сохранены; latency не измерялась.

- [Выполненный notebook](zero-init.ipynb).
- [Основной подробный отчёт](runs/reproduction-smoke/report.md), CSV/JSON — рядом.
- [Альтернативы](runs/alternatives-smoke/report.md).
- [Figure 5](runs/reproduction-smoke/plots/figure5-resnet18.png), [Figure 6](runs/reproduction-smoke/plots/figure6-resnet18.png), [Figure 8](runs/reproduction-smoke/plots/figure8-resnet18.png); PDF находятся рядом.
- [CPU benchmark](runs/local-benchmark/benchmark.json).
- [Лог тестов](runs/final-tests.log).

## Измеренная стоимость и ограничения

Платформа: macOS arm64, Python3.12.9, torch2.12.0, torchvision0.27.0, NumPy2.2.4; CPU4 threads, strict FP32, TF32=False. CUDA отсутствует. Данные CIFAR-10 доступны локально и проверены по MD5.

Отдельный benchmark полного batch128: **4.539 s /3 training batches**, **1.553 s /3 test batches**, **.065 s** на три stable-rank SVD; process peak RSS **1228 MB**. Экстраполяция: около **632 s/epoch**, **1053 h** на основную серию 30×200 epochs, **3632 s** на шесть400-step запусков Figure8. Это приблизительная оценка короткого CPU-замера, не фактически потраченное время полной серии; checkpoint I/O исключён. Сырые selected kernels: около1.24 GB/run, 37.3 GB/30runs, плюс checkpoints.

Полное обучение локально не запущено из-за измеренной стоимости. Точные команды для GPU, resume, Figure8, compression и анализа — в [README.md](README.md). Отсутствующие результаты не генерировались.

Малая разница accuracy не доказывает превосходство. Низкий stable rank не доказывает GLRL. Архитектура, preprocessing, scheduler, norm type Figure8, список conv для Figure6 и точный Tucker-layer содержат явно обозначенные реконструкции. ConstNet* зависит от CUDA/cuDNN nondeterminism и требует отдельной проверки. Figure4/depth, ResNet-50, MNIST и Transformer исключены.
