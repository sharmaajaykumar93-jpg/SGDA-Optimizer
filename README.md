# SGDA Optimizer

This repository contains the official Python/PyTorch implementation of the experiments reported in the manuscript:

**An Adaptive Stochastic Gradient Descent Method with Normalized Momentum**

## Description

SGDA is a first-order stochastic optimizer designed for noisy and nonstationary learning environments. It uses normalized gradient differences, bounded momentum, and an adaptive step-size mechanism to improve training stability while keeping low computational and memory cost.

## Repository Contents

```text
SGDA-Optimizer/
├── README.md
├── requirements.txt
├── optimizers/
│   └── sgda.py
├── experiments/
│   ├── usps_lenet5.py
│   ├── rotating_mnist_convnet.py
│   └── cifar10_resnet18.py
└── results/
