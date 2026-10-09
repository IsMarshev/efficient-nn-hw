# ZerO initialization

В [`zero-init.ipynb`](zero-init.ipynb) собраны ResNet-18/50, инициализация ZerO и сравнение с Kaiming/Xavier на CIFAR-10.

## Запуск

1. Откройте ноутбук с рабочим каталогом `hw2` и выполните ячейки последовательно. Первая ячейка устанавливает зависимости.
2. Оставьте `PRESET = "smoke"` для проверки: ResNet-18, три метода, один seed, одна эпоха на подмножестве данных.
3. Для длительного эксперимента смените `PRESET` на `"full"`: обе сети, три seed, 200 эпох. Конфигурацию можно менять через `dataclasses.replace`.
4. Ячейка `run_experiments(config)` скачивает CIFAR-10 и запускает обучение; следующая строит графики и таблицу.

Устройство выбирается автоматически: CUDA → MPS → CPU. Можно задать `device="cpu"` или `device="cuda"`. Для запуска на macOS в ноутбуке по умолчанию используется `num_workers=0`.

## Что реализовано

- `init="zero"`: структурированная инициализация по Algorithms 1–2 [статьи](https://arxiv.org/pdf/2110.12661), с нулевой последней свёрткой каждого остаточного блока.
- `init="identity"`: ablation без Hadamard; `init="zeros"`: полное обнуление Conv/Linear. Сохранены случайные методы, пользовательская функция и `zero_init_residual`.
- CIFAR stem 3×3, аугментации train, отдельные train/validation/test, SGD, linear warmup + cosine decay.
- Loss/accuracy, stable rank трёх свёрток, сравнение среднего и разброса по seed.
- Проверка детерминированности начальных весов и обновления остаточных ветвей обеих сетей.

Это учебная адаптация с BatchNorm и собственным validation split / cosine schedule, а не точное воспроизведение всех таблиц статьи. Короткий smoke-прогон проверяет код, но не качество метода. ImageNet, обучение без BN и pruning здесь не реализованы.

## Результаты

Относительно рабочего каталога ядра создаются `data/` и `runs/zero-init/<timestamp>/`. В каталоге каждого метода и seed лежат:

- `config.json` — настройки, устройство, размеры выборок и версия PyTorch;
- `history.json` / `history.csv` — метрики по эпохам, включая исходное состояние;
- `best.pt` — веса checkpoint с лучшей validation accuracy (не checkpoint для возобновления оптимизатора);
- `summary.json` — test-метрики выбранной модели.

Общая папка содержит `summary.csv`, `aggregate.csv` и PNG-графики. Для просмотра сохранённого запуска вызовите `plot_results(Path("runs/zero-init/<timestamp>"))`.
