# Протокол ResNet-18 / CIFAR-10

Источник P: [ZerO, arXiv:2110.12661v3, 4 ноября 2022](https://arxiv.org/pdf/2110.12661v3), 18 страниц, включая приложения A–C. Источник C: [официальный репозиторий](https://github.com/jiaweizzhao/ZerO-initialization/tree/cf2bf5d7d967958b3bb232755b288d97239896a8), commit `cf2bf5d7d967958b3bb232755b288d97239896a8`. Локальные копии и SHA256: `sources/manifest.json`.

**Точное воспроизведение CIFAR-протокола невозможно по опубликованным артефактам.** В C только README, LICENSE и `example_mnist.ipynb`. Нет ResNet, CIFAR training, альтернативных residual-блоков или compression scripts. Не переносим MNIST batch=64, StepLR и SGD без momentum в CIFAR. Ниже P = явно в статье, C = официальный код, R = наша реконструкция, D = дополнительная диагностика. Конфигурации заморожены до измерений.

| Эксперимент | Источник | Архитектура | Настройки | Метрики | Соответствие |
|---|---|---|---|---|---|
| Table 2, воспроизводимость | P §4, Table 2 | ResNet-18/CIFAR-10 с BN | P: LR .1, momentum .9, WD 1e-4, warmup 10 epochs, 10 runs; R: batch 128, 200 epochs, linear warmup+cosine | last top-1 error, sample std, 95% Student-t CI, paired differences | Частичная реконструкция |
| Table 3, ResNet-18 | P §4, первичные работы ниже | BN ZerO; явные варианты ReZero/Fixup/SkipInit/ConstNet | общий SGD-протокол выше; R: адаптация блоков к [2,2,2,2]; ConstNet* только nondeterministic CUDA | last error; опубликованные 5.13/5.20/5.17/5.26/72.39/5.41 отдельно | Реконструкция, не авторская ResNet-18 |
| Figure 5, верхние 3 панели | P caption, §5 | тот же ResNet-18 | первые conv групп 2/3/4: `layer2.0.conv1`, `layer3.0.conv1`, `layer4.0.conv1`; initial и каждая эпоха | stable rank, сырые веса и CSV | Слои соответствуют описанию; развёртка R |
| Figure 6, слева | P §5, caption | конечные модели Table 2 | R: stem + 16 residual conv, исключить projection; развёртка `[Cout,Cin*kH*kW]` | stable rank | Алгоритм панели не раскрыт отдельно; stable rank выбран по §5, не numerical rank MNIST |
| Figure 6, центр | P §5 | каждый исходный last checkpoint независимо | P: magnitude внутри каждого слоя; R: Conv+Linear, projection включены, BN/bias исключены; keep .9,.85,...,.5; без fine-tuning | loss, accuracy, retained slots и actual nonzero fraction | Часть набора слоёв/сетки неизвестна |
| Figure 6, справа | P §5/caption | выбранный Conv 512→512 | R: `layer4.1.conv2`, равные `(r_out,r_in)`, r=60,50,40,30,25,20,15,10,5; HOSVD+5 HOOI; 1×1→3×3→1×1 | accuracy, loss, параметры, MACs | Точный слой, алгоритм выбора рангов неизвестны |
| Figure 8, две панели | P Appendix C | ResNet-18/CIFAR-10, `conv1.weight` | P: 400 iterations, initial LR .01/.1; R: L2/Frobenius norm, warmup линейный на 10 полных эпох, общий cosine далее; FP32 | gradient norm до step, LR, loss, batch SHA256 | Тип нормы/точное расписание R |
| ResNet-18 без BN | P §3 лишь общее правило; D | scalar biases/multiplier вместо BN | отдельный `zero_no_bn`, общий протокол | error, gradients | Дополнительная проверка, не Figure 4 |

## Архитектура и инициализация

R: CIFAR stem 3×3/stride1/pad1, 3→64; Conv→BN→ReLU, без maxpool. 4 группы [2,2,2,2], channels 64/128/256/512. BasicBlock: Conv3×3→BN→ReLU→Conv3×3→BN→add→ReLU. Первая conv групп 2–4 stride2; shortcut 1×1/stride2 + BN. AdaptiveAvgPool 1×1, Linear512→10 с bias. BN eps=1e-5, momentum=.1 (PyTorch defaults, R). Это явная реконструкция, **не точная архитектура авторов**. CLI `inspect` сохраняет полную схему, формы каждой операции и число параметров.

P Algorithms 1–2: если out>in, Sylvester Hadamard order N=2^ceil(log2(out)), crop [:out,:in], множитель `sqrt(2/N)`; иначе partial identity. Conv: только центр, все остальные элементы 0. Последняя Conv residual-ветви нулевая, BN gamma=1/beta=0. Stem, projections и classifier также инициализированы, biases=0. Значения ±sqrt(2/N), а не буквально только {0,1}.

**Расхождение P/C:** C делит Hadamard на sqrt(N), P на sqrt(N/2). Выбираем формулу P, не «исправляем» на привычную ортонормировку. Для полного квадратного H коэффициент из P даёт WᵀW=2I, хотя текст называет transform orthonormal. Это также фиксируем как несогласованность текста/формулы. C `compute_rank` использует numerical rank tol=1e-4 для MNIST и вычитает identity; это не алгоритм Figure 5. В наших conv stable rank измеряется на полном ядре без вычитания identity; §5 трактует conv как residual компонент conv+I.

R: Kaiming normal fan_out/ReLU для Conv, fan_in/linear для FC; Xavier normal gain1 для Conv/FC. ZerO/случайные методы имеют одну BN-архитектуру; случайные методы не обнуляют conv2. WD на всех обучаемых параметрах, включая BN и scalars, одинаково для методов (авторское исключение параметров неизвестно).

## Альтернативные методы

[Fixup, первичный репозиторий](https://github.com/hongyi-zhang/Fixup/tree/f3fa449fbb8906ccb7143cea822f8b964b3d9258), `imagenet/models/fixup_resnet_imagenet.py`: адаптация BasicBlock, четыре scalar bias (до conv1, ReLU1, conv2 и ReLU2), multiplier=1. BN нет; conv1 random Kaiming fan_out×8^-1/2, conv2=0, FC=0. Stem и projection random Kaiming (stem задан явно вместо оставленного default). Scalar biases около stem/FC. Только stem заменён CIFAR-вариантом. Это самостоятельная реализация правил, не копия обучающего кода с MixUp.

[ReZero](https://arxiv.org/pdf/2003.04887), правило x+αF(x), α=0. R: убираем BN из stem/blocks/shortcut, random Kaiming, обучаемый α в каждом блоке; post-activation сохранена для сравнения на ResNet-18. Из-за post-ReLU/проекций вся сеть не буквально identity.

[SkipInit](https://arxiv.org/pdf/2002.10444), Figure 1 и Appendix F: BN нет, α=0, random He fan_in; R: добавлен scalar bias после сложения перед final ReLU (вариант для ResNet-V1 Appendix F), conv biases=0. Не включаем регуляризованный dropout. Это отдельный блок, не alias ReZero.

[ConstNet, первичная работа](https://proceedings.mlr.press/v119/blumenfeld20a.html), [код](https://github.com/yanivbl6/BeyondSigProp/tree/92c4c8abb60b699ba17d22a80419e25938ffb075), `model.py`: stem/projection center=1/Cin, residual conv=0, classifier=0, BN=1/0. R: адаптация к двухсвёрточному BasicBlock [2,2,2,2], conv1 внутри residual задан averaging и conv2=0 (код оригинала использует одну residual Conv на блок). Сохраняет начальную симметрию каналов; не претендует на точную архитектуру оригинала. ConstNet выполняется строго детерминированно. ConstNet* те же начальные веса, cuDNN benchmark=True, deterministic=False, deterministic algorithms=False на CUDA; без искусственного шума/dropout. Наличие nondeterministic kernels на современной платформе не гарантирует повторение старого CUDA10/cuDNN7 эффекта. CPU/MPS не подменяют ConstNet*.

## Данные, checkpoint и RNG

R: исходные CIFAR-10 50k train /10k test, crop32/pad4, random horizontal flip .5, float /255, mean (.4914,.4822,.4465), std (.2470,.2435,.2616). Архив torchvision с MD5. Pretrained/MixUp/CutMix/smoothing отсутствуют. Test мониторится, гиперпараметры по нему не выбираются. Публикуем last checkpoint отдельно от best-by-observed-test (описательная верхняя оценка, не основной результат); критерий авторов не раскрыт.

Seeds фиксированы 0..9 для каждого метода. Инициализация внутри fork_rng с init_seed; порядок данных отдельный Generator с domain seed; crop/flip имеют собственный Generator, seed=f(run_seed,epoch,original_index), не зависят от workers или расхода RNG модели. Figure 8 сохраняет hash фактического batch и сравнивает hashes всех шести запусков. Resume на границе эпох восстанавливает optimizer, scheduler step, Python/NumPy/Torch/CUDA RNG, историю и лучший checkpoint; сырые rank snapshots включают epoch0 и каждую эпоху.

Strict: deterministic_algorithms=True, cuDNN deterministic=True/benchmark=False, TF32=False, FP32. Original: nondeterministic CUDA разрешена, cuDNN benchmark=True; прочие настройки FP32/TF32=False. За исключением ConstNet*, точный режим авторов не известен, название original означает разрешённый режим, а не подтверждённую аппаратную конфигурацию. AMP BF16 — только отдельная явно указанная дополнительная конфигурация.

## Статистика и ограничения

Sample std ddof=1; 95% CI mean ± t_(.975,n-1)*s/sqrt(n), n<2 → undefined. Парные разности по одинаковым seed; отдельно список run errors, variance/std ratio. Пропущенные seeds/несовместимые configs не объединяются молча. Small mean differences не доказывают превосходство; CI описывают sampling variability, не устраняют protocol mismatch. Smoke и частично обученные модели не сравниваются с ориентирами Table 2 как окончательные результаты. Stable rank не равен numerical или Tucker rank; уменьшение stable rank не доказывает GLRL.

Figure 4 с изменением глубины невозможно воспроизвести на одной ResNet-18. Figure 3/7 MNIST, Figure 5 нижний ряд, ResNet-50 и Transformer исключены. Перечень невыполненных полных запусков формируется отчётом из существующих результатов, отсутствующие кривые не дорисовываются.
