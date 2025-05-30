from config import EMOJI
from Pattern import Pattern
import matplotlib.pyplot as plt
import numpy as np
import torch
import utils

PT0 = Pattern(EMOJI)


data = PT0.data

data = data * utils.perturb(1).numpy()[0, :4, ...]
data = np.transpose(data, (2, 1, 0))
plt.imshow(data)
plt.show()
