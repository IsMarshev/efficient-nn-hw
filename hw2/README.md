# ZerO: запуск на H100 за двухчасовую аренду

Код моделей, инициализации, обучения и графиков находится в [`zero-init.ipynb`](zero-init.ipynb). [`run_zero.py`](run_zero.py) читает помеченные ячейки этого же ноутбука: отдельной копии модели нет.

## Запуск на GPU

Загрузите папку `hw2` в GPU-среду с установленными **CUDA-версиями PyTorch и torchvision** (рекомендуется готовый PyTorch-образ провайдера). Рабочий каталог — `hw2`.

```bash
nvidia-smi
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
python -m pip install -r requirements.txt
python run_zero.py --preset h100_2h --device cuda > training.log 2>&1
```

Для просмотра из другого терминала: `tail -f training.log`. CUDA PyTorch из образа переустанавливать не нужно. `requirements.txt` устанавливает только библиотеки анализа. Предусмотрено 8 загрузчиков данных; при малом числе CPU задайте `--num-workers 4` **до начала запуска**.

`h100_2h`:

- ResNet-18 / CIFAR-10, ZerO и Kaiming, один seed;
- цель — 200 эпох на каждый метод, все 50 000 train и 10 000 test;
- SGD: LR 0.1, momentum 0.9, weight decay 1e-4;
- 10 эпох warmup от 0.01, далее cosine;
- BF16 autocast, FP32-параметры/SGD-состояние, channels-last;
- 110 минут от старта скрипта; начальный замер и загрузка входят в бюджет;
- замер скорости автоматически выбирает **одинаковое** число эпох для обоих методов, максимум 200 и минимум 50. Если минимум не помещается, обучение не начинается;
- каждая полная эпоха сохраняет checkpoint; при исчерпании времени есть возможность продолжить.

Это оценка времени с запасом, а не гарантия производительности конкретной аренды. Фактический план записывается в `runtime_plan.json`; `run_config.json` определяет реальное число эпох и расписание. Для фиксированных 200 эпох используйте `--no-auto-fit`: если времени не хватит, потребуется resume. Не меняйте число эпох после начала обучения — это изменит cosine schedule.

## Соответствие статье

Используется [arXiv:2110.12661v3](https://arxiv.org/pdf/2110.12661v3), Algorithms 1–2. ZerO задаёт частичные identity/Hadamard-преобразования в центре ядер и обнуляет последнюю Conv остаточного блока; BatchNorm γ=1, β=0. Полное обнуление сети и `zero_init_residual` — другие режимы.

В [официальном репозитории](https://github.com/jiaweizzhao/ZerO-initialization) опубликован только MNIST-пример, без рецепта ResNet. Batch size, decay, часть архитектурных и preprocessing-деталей не раскрыты. Наши batch=128, cosine, CIFAR-stem 3×3, crop/flip, нормализация и развёртка ядра для stable rank зафиксированы явно. Формула нормировки Hadamard берётся из PDF; она отличается от MNIST-примера авторов.

Полный train используется без дополнительного holdout. Test мониторится, но **checkpoint всегда последней эпохи**: нет выбора максимума по test. Гиперпараметры не подбираются под test-метрику. В `aggregate.csv` опубликованные ориентиры и измерения находятся в разных колонках.

Похожая accuracy — проверяемая цель, а не обещание. Два часа не рассчитаны на 10 seed и ImageNet/ResNet-50. Для полного сравнения с оценкой разброса есть `--preset paper_cifar` (3 метода × 10 seed, FP32, без лимита); `cifar_both` добавляет ResNet-50 на CIFAR-10 и не воспроизводит её ImageNet-эксперимент.

## Результаты и продолжение

Скрипт печатает каталог `runs/zero-init/<timestamp>/`. **До окончания аренды скачайте весь этот каталог и `training.log`.** Внутри:

- `runtime_plan.json`, `run_config.json` — замер скорости и согласованный бюджет;
- `<model>-<method>-seed<N>/history.json` / `.csv` — реальные метрики по эпохам;
- `last.pt` — веса, momentum оптимизатора, конфигурация, номер полной эпохи;
- `summary.json` — итог только завершённого запуска; `status.json` показывает остановку по времени;
- `aggregate.csv` — сравнение измеренного test error с отдельным ориентиром статьи;
- `figure5-stable-rank.png/.pdf`, `resnet18-training-curves.png/.pdf` — графики из логов.

```bash
python run_zero.py --resume runs/zero-init/<timestamp> --device cuda --budget-minutes 110
python run_zero.py --plot-only runs/zero-init/<timestamp>
```

Resume сохраняет старое расписание. Незаконченная эпоха пересчитывается из последней полной; seed аугментаций и перестановки зависит от номера эпохи. При остановке по бюджету CLI возвращает код 2, при полном завершении — 0.

## Локальная проверка

```bash
python -m unittest -v test_zero
python run_zero.py --preset smoke --device cpu
```

Тесты проверяют формулу ZerO, остаточные блоки, warmup и идентичность непрерывного / возобновлённого обучения на CPU. Smoke действительно обучает на 512 изображениях CIFAR-10 две эпохи, но его качество не сравнимо со статьёй. H100/CUDA/BF16 должны быть проверены на арендованной машине начальным замером.
