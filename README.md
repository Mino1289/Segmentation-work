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

## Usage

Not yet