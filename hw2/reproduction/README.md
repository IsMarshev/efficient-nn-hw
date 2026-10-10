# Воспроизведение ZerO Initialization

Рабочая папка — `hw2/reproduction`. Обучение реализовано в Python-модулях, notebooks читают сохранённые результаты из `runs/`.

```bash
python -m pip install -r requirements.txt
# CIFAR-10: три метода × один seed, 200 эпох.
python run_zero.py seeds --config configs/main.json --seeds 0 --device cuda --root runs/cifar-new
# Figure 8: шесть запусков по 400 шагов.
python run_zero.py gradients --config configs/figure8.json --device cuda --root runs/figure8-new --steps 400
# Figure 6: сжатие checkpoints нового обучения.
python run_zero.py compression --config configs/main.json --device cuda --root runs/cifar-new
# MNIST: для остальных методов используйте Random или Partial_Identity.
python run_mnist.py --method ZerO --width 2048 --root runs/mnist-new
python -m unittest -v test_zero test_figure6
```

Анализ: `zero-init.ipynb` (CIFAR) и `mlp_ZeRO.ipynb` (MNIST). Для левой панели Figure 3 запустите Partial_Identity с ширинами 256/512/1024/2048, сохраняя каждый запуск отдельно. Датасеты скачиваются автоматически. Сырые истории ранга и checkpoints прежних запусков не сохранены; их панели читаются из экспортированных изображений. Используйте новые каталоги `--root`, чтобы сохранить исходные результаты.
