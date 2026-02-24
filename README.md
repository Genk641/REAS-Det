# REAS-Det

Official open-source implementation for the paper:

**A Multi-Strategy Framework for Enhancing Shatian Pomelo Detection in Real-World Orchards**  

[Paper (arXiv:2510.09948)](https://arxiv.org/abs/2510.09948) | [PDF](https://arxiv.org/pdf/2510.09948)

## Overview

Shatian pomelo detection in real orchards is challenging due to illumination variation, tone shift across devices, scale changes, and heavy occlusion.

This project provides the implementation of **REAS-Det**, a multi-strategy detector built on a modified Ultralytics/YOLO codebase. The repository includes:

- **GSV-Conv** (Global-Selective Visibility Convolution)
- **C3RFEM**
- **MultiSEAM**
- **Soft-NMS** integration (customized post-processing)

## Repository Structure

Key files in this repository:

- `network/REAS-Det.yaml`: REAS-Det model configuration
- `ultralytics/nn/extra_modules/GSVConv.py`: GSV-Conv implementation
- `ultralytics/nn/extra_modules/SEAM_Multi.py`: MultiSEAM-related modules
- `ultralytics/nn/tasks.py`: custom module registration / model parsing integration
- `ultralytics/utils/ops.py`: custom post-processing (e.g., Soft-NMS modifications)
- `train.py`, `main.py`: example training entry scripts

## Environment Setup

This repository is a **modified Ultralytics codebase**. Please prepare a Python environment first.

### 1. Create environment

```bash
conda create -n reasdet python=3.10 -y
conda activate reasdet
```

### 2. Install PyTorch (pip, CUDA 11.8)

Recommended version for this project:

- `torch==2.0.1+cu118`

Install with `pip`:

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install common dependencies

```bash
pip install ultralytics einops opencv-python pyyaml scipy matplotlib tqdm
```

### 4. Install Mamba dependency (required for GSV-Conv)

`GSVConv.py` uses `mamba_ssm` (`selective_scan_fn`).

Please download and install the following packages manually:

- `mamba_ssm==1.0.1` from: https://github.com/state-spaces/mamba/releases?page=3
- `causal-conv1d==1.0.0` from: https://github.com/Dao-AILab/causal-conv1d/releases?page=4

After downloading the appropriate wheel files for your Python / CUDA environment, install them with `pip`:

```bash
pip install causal_conv1d-1.0.0-*.whl
pip install mamba_ssm-1.0.1-*.whl
```

If installation fails, check Python / CUDA / PyTorch version compatibility of the wheel files.

## Training

You can train REAS-Det by using the modified Ultralytics API.

### Option A: Use a Python script (recommended for custom experiments)

Example (adapt to your dataset path and class count):

```python
from ultralytics import YOLO

model = YOLO("network/REAS-Det.yaml")
model.train(
    data="path/to/your_dataset.yaml",
    imgsz=640,
    epochs=100,
    batch=16,
    workers=8,
    device="0",
    project="runs/train",
    name="REAS-Det",
)
```

### Option B: Run the provided scripts

This repo includes example scripts:

- `train.py`
- `main.py`

Before running them, update the dataset config path (e.g., `data=...`) to your local environment.

```bash
python train.py
```

or

```bash
python main.py
```

## Validation / Inference

### Validation

```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")
metrics = model.val(data="path/to/your_dataset.yaml")
print(metrics)
```

### Inference

```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")
results = model("path/to/image_or_folder")
```

## Dataset

The paper introduces **STP-AgriData** (real orchard + curated web images with illumination augmentation).

- If/when the dataset is released, please place the dataset files locally and prepare a YOLO-format dataset YAML.
- Update the `nc` field in `network/REAS-Det.yaml` and your dataset YAML according to your task.

Example dataset YAML (YOLO format):

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

names:
  0: shatian_pomelo
```

## Model Configuration

The main REAS-Det architecture file is:

- `network/REAS-Det.yaml`

If you change the number of classes, update:

- `nc` in `network/REAS-Det.yaml`
- `names` / class definitions in your dataset YAML

## Notes

- This codebase modifies Ultralytics internals; mixing it with an external `ultralytics` package version may cause incompatibilities.
- `GSVConv` depends on `mamba-ssm`; ensure the dependency is correctly installed before training/inference.
- Some script paths in `train.py` / `main.py` are local examples and should be replaced with your own paths.

## Citation

If you find this repository useful, please cite the paper:

```bibtex
@article{wang2025reasdet,
  title={A Multi-Strategy Framework for Enhancing Shatian Pomelo Detection in Real-World Orchards},
  author={Wang, Pan and Hu, Yihao and Bai, Xiaodong and Yang, Jingchu and Zhou, Leyi and Yang, Aiping and Li, Xiangxiang and Ding, Meiping and Yao, Jianguo},
  journal={arXiv preprint arXiv:2510.09948},
  year={2025}
}
```

## Acknowledgment

This project is implemented on top of the Ultralytics/YOLO framework and extends it with custom modules for robust orchard fruit detection.
