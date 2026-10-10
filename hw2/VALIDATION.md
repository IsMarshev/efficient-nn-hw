# Выполненные проверки

10 октября 2026, локальная CPU-платформа macOS arm64.

```bash
python -m unittest -v test_zero test_figure6
python run_zero.py smoke --device cpu --root runs/reproduction-smoke
python run_zero.py gradients --config configs/smoke.json --root runs/figure8-smoke --steps 8
python run_zero.py compression --config configs/smoke.json --root runs/reproduction-smoke --test-limit 64 --keep .9 .7 .5 --ranks 20 5
python run_zero.py alternatives --config configs/smoke.json --methods rezero fixup skipinit constnet zero_no_bn --root runs/alternatives-smoke
python run_zero.py benchmark --config configs/main.json --device cpu --root runs/local-benchmark
```

10 tests: initialization all parameters; stable ranks; scheduler/statistics; explicit alternative blocks and finite updates; exact resume; actual CIFAR RNG isolation and worker count; pruning independent points; Tucker full reconstruction/channel ranks; full executable Tucker stride/padding/dilation/bias; reduced unequal channel ranks vs equivalent dense kernel.

Notebook исполнен через nbclient; все analysis cells прошли, 8 research panels и3 дополнительных training panels сохранены. Notebook не запускает обучение. Smoke gradients: 48 rows, шесть групп, batch hashes совпадают. Full-rank Tucker на всех трёх 512-channel kernels прошёл scale-aware FP32 tolerance; relative Frobenius errors <1.9e-6.

Полные 200-epoch эксперименты, 400-step Figure8, полный test compression и CUDA/ConstNet* **не выполнены**. Это перечислено в REPORT.md. CPU benchmark и реальные результаты находятся в `runs/`; их отсутствие при чистом checkout не заменяется фиктивными данными.
