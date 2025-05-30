import torch

EMOJI = "mage.png"

GRID_SIZE = 100

PATTERN_SIZE = 64

if torch.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else: 
    DEVICE = "cpu"
    print("Warning : training is using CPU")

