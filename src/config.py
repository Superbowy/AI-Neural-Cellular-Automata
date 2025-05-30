import torch

EMOJI = "../emojis/lizarg.png"

GRID_SIZE = 64

PATTERN_SIZE = 40

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.mps.is_available():
    DEVICE = "mps"
else: 
    DEVICE = "cpu"
    print("Warning : training is using CPU")

