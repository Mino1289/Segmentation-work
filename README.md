# Segmentation work

Implémentation PyTorch de modèles de segmentation sur ADE20K :

- **SAM** (ViT-H) — segmentation interactive par prompts
- **DINOv2 / DINOv3** (ViT-Base) — linear probe sur ADE20K
- **ConvNeXtV2 + UPerNet** — segmentation sémantique end-to-end

Dataset : [merve/scene_parse_150](https://huggingface.co/datasets/merve/scene_parse_150) (150 classes, splits train / validation / test).

## Installation

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

Note : installez `torch` et `torchvision` séparément si besoin pour votre backend GPU : https://pytorch.org/get-started/locally/

Le fichier `requirements.txt` inclut `gradio` pour la démo SAM.

## Structure du projet

```
models/          # Architectures (SAM, DINOv2/v3, ConvNeXtV2, UPerNet)
dataset/         # Chargement ADE20K
eval/            # Benchmark et utilitaires d'évaluation (model_registry.py, viz.py, data.py)
demo/            # Démo interactive SAM (Gradio)
notebooks/       # Visualisation comparative des modèles
weights/         # Checkpoints entraînés
train_*.py       # Scripts d'entraînement
```

## Entraînement

Placez les checkpoints dans `weights/` après entraînement.

```bash
# DINOv2 linear probe (518px, 60 epochs)
python train_linear_dinov2.py
# → weights/linear_dinov2_base.pth

# DINOv3 linear probe (512px, 60 epochs)
python train_linear_dinov3.py
# → weights/linear_dinov3_base.pth

# ConvNeXtV2 + UPerNet (512px, 128 epochs)
python train_upernet.py
# → weights/upernet_convnextv2_base.pth
```

## Données ADE20K (notebooks / démo)

Chargement du split **test** ADE20K et images personnelles via `eval/data.py` :

```python
from eval.data import get_ade20k_sample, load_personal_image, list_test_indices
image, mask = get_ade20k_sample(42, split="test")
```

## Benchmark ADE20K

Évalue les modèles sémantiques sur le split **validation** (mIoU, pixel accuracy, Boundary IoU, paramètres, GFLOPs).

```bash
python eval/benchmark_ade20k.py \
  --models dinov2_linear dinov3_linear upernet \
  --split validation \
  --batch-size 8 \
  --output results/benchmark_ade20k.csv
```

Override d'un checkpoint :

```bash
python eval/benchmark_ade20k.py \
  --checkpoint dinov2_linear=weights/mon_checkpoint.pth
```

Résultats exportés en CSV et JSON dans `results/`.

| Métrique | Description |
|----------|-------------|
| pixel_acc | Proportion de pixels correctement classés |
| mIoU | Moyenne IoU sur les 150 classes |
| boundary_iou | IoU sur les frontières d'objets |
| params_M | Nombre total de paramètres (millions) |
| gflops | MACs / 10⁹ à la résolution d'inférence |

## Visualisation

Utilitaires dans `eval/viz.py` : palette ADE20K, overlay aligné (resize+center crop),
légende des classes présentes (`eval/ade20k_classes.py`).

Notebooks dans `notebooks/` — images du split **test** ADE20K et images personnelles (`dataset/dog.jpg`, `dataset/rue.jpg`).

| Notebook | Description |
|----------|-------------|
| `compare_segmentation.ipynb` | Grille 3×3 par modèle (image \| overlay \| masque) |
| `visualize_ade20k_samples.ipynb` | Ground truth + prédictions sur le test ADE20K |

Lancez depuis la racine du projet ou depuis `notebooks/` (les notebooks ajustent `sys.path` automatiquement).

## Démo SAM (Gradio)

Segmentation interactive par clics (foreground / background).

**Prérequis** : télécharger le checkpoint Meta SAM ViT-H et le placer dans `weights/sam_vit_h_4b8939.pth`.

```bash
python demo/sam_app.py
# Lien public optionnel :
python demo/sam_app.py --share
```

Fonctionnalités :
- Chargement d'images du **test ADE20K** (slider d'index)
- Upload d'images personnelles
- Clics foreground (objet) / background (exclusion)
- Affichage overlay, masque binaire et 3 candidats multimask avec scores IoU

## Checkpoints

| Fichier | Modèle | Source |
|---------|--------|--------|
| `weights/linear_dinov2_base.pth` | DINOv2 linear | `train_linear_dinov2.py` |
| `weights/linear_dinov3_base.pth` | DINOv3 linear | `train_linear_dinov3.py` |
| `weights/upernet_convnextv2_base.pth` | ConvNeXtV2 + UPerNet | `train_upernet.py` |
| `weights/sam_vit_h_4b8939.pth` | SAM ViT-H | [Meta SAM](https://github.com/facebookresearch/segment-anything#model-checkpoints) |
