# Segmentation work

PyTorch Implementation of:
- SAM
- DINO
- ConvNeXt

Later, we'll try using the weights from huggingface and timm, to see if we can get the same results as the original implementations. 

## Installation

We recommend using `uv` as it's very fast and easy to use for creating a virtual environment for this project.
```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

Note: You may need to install `torch` and `torchvision` separately, to get the appropriate gpu backend depending on your system. You can find the appropriate versions for your system at https://pytorch.org/get-started/locally/.

## Structure du projet

```
eval/            # Registry modèles, benchmark, visualisation (model_registry.py, data.py)
models/          # Architectures (SAM, DINOv2/v3, ConvNeXtV2, UPerNet)
dataset/         # Chargement ADE20K
weights/         # Checkpoints entraînés
train_*.py       # Scripts d'entraînement
```

## Données ADE20K (notebooks / démo)

Chargement du split **test** ADE20K et images personnelles via `eval/data.py` :

```python
from eval.data import get_ade20k_sample, load_personal_image, list_test_indices
image, mask = get_ade20k_sample(42, split="test")
```

## Visualisation

Utilitaires dans `eval/viz.py` : palette ADE20K, overlay aligné (resize+center crop),
légende des classes présentes (`eval/ade20k_classes.py`).

## Usage

Not yet