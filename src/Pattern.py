import numpy as np
from config import GRID_SIZE, PATTERN_SIZE
from PIL import Image

class Pattern:

    def __init__(self, path):
        ref = Image.open(path)
        ref.thumbnail((PATTERN_SIZE, PATTERN_SIZE), Image.LANCZOS)
        ref = np.array(ref) / 255
        padding =( GRID_SIZE - PATTERN_SIZE) // 2
        ref = np.pad(ref, ((padding, padding), (padding, padding), (0, 0)))
        ref = np.transpose(ref, (2, 1, 0))
        self.data = ref

    def shape(self):
        return self.data.shape

    def as_image(self):
        return np.transpose(self.data, (2, 1, 0))
