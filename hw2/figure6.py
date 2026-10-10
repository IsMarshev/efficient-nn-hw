"""Compatibility imports for compression; use run_zero.py compression for measurements.
Plotting is performed in zero-init.ipynb.
"""
from zero_experiments.compression import magnitude_prune, tucker2_approx, channel_bases
from zero_experiments.cli import main

if __name__ == '__main__':
    main(['compression', *__import__('sys').argv[1:]])
