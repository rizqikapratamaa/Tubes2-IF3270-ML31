# 𝙲𝙽𝙽 & 𝚁𝙽𝙽 𝙵𝚛𝚘𝚖 𝚂𝚌𝚛𝚊𝚝𝚌𝚑

> **Tugas Besar IF3270 - Machine Learning**

> CNN and RNN are two of the most popular deep learning architectures, designed respectively for spatial and sequential data. This project implements CNN, RNN (SimpleRNN), and LSTM from scratch, including forward propagation and performance analysis experiments based on various model configurations.

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Setting Up](#setting-up)
4. [Bonus](#bonus-🤑)
5. [Acknowledgements](#acknowledgements)

## Overview

This project manually implements three deep learning architectures (without TensorFlow/PyTorch):
- Convolutional Neural Network (CNN) for image classification (CIFAR-10)
- Recurrent Neural Network (SimpleRNN) for sentiment analysis (NusaX dataset)
- Long Short-Term Memory (LSTM) as an enhancement of RNN

Key features supported:
- Modular layer architecture (Conv2D, Pooling, RNN, LSTM, Dense, Embedding)
- Manual forward propagation using NumPy
- Loading weights from trained Keras models
- Experiments on effects of number of layers, filter/unit counts, layer direction, etc.
- Performance comparison between manual implementation and Keras

## Important Notice

> [!IMPORTANT]\
> This project uses models trained with Keras, but all forward propagation is manually implemented using various libraries. Make sure to install all dependencies before running the code. See the [Requirements](#requirements) section for details.

## Requirements

- `Python`
- `joblib==1.4.2`
- `numpy==2.2.3`
- `scikit-learn==1.6.1`
- `scipy==1.15.2`
- `threadpoolctl==3.5.0`
- `tqdm==4.67.1`
- `matplotlib==3.10.1`
- `networkx==3.4.2`

## Setting Up

> [!NOTE]\
> This setup is for WSL. If you are developing on Windows, create your own virtual environment.

<details>
<summary>:eyes: Get Started</summary>  

#### Clone the Repository:

```sh
 git clone https://github.com/rizqikapratamaa/Tubes1_ML_51.git
 cd TUBES1_ML_51
```

#### Create new env

```sh
 python3 -m venv env_tubes
 source env_tubes/bin/activate
```

#### Install requirements

```sh
 pip install -r requirement.txt
```

#### Run the program

```sh
 python3 main.py
```

#### After finishing, exit from venv

```sh
 deactivate
```

</details>

## Bonus 🤑

> [!NOTE]\
> Backward propagation was implemented but not used in main experiments; batch inference was used during forward propagation, and backward propagation is indirectly applied during training.

<summary>1. Backward propagation functions were implemented from scratch for LSTM. (13522126)</summary>
<summary>2. The forward propagation implementation supports batch inference, allowing the model to process multiple inputs in a single forward pass. (13522126)</summary>

## Developers

| Name                  | NIM      | Connect                                                |
| --------------------- | -------- | ------------------------------------------------------ |
| Rizqika Mulia Pratama | 13522126 | [@rizqikapratamaa](https://github.com/rizqikapratamaa) |
| Attara Majesta Ayub   | 13522139 | [@attaramajesta](https://github.com/attaramajesta)     |
| Ikhwan Al Hakim       | 13522147 | [@Nerggg](https://github.com/Nerggg)                   |

## Acknowledgements

- Machine Learning Course Lecturer, Bandung Institute of Technology, 2025
- Machine Learning Teaching Assistants, Bandung Institute of Technology, 2025
