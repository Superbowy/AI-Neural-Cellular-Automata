import numpy as np
from PIL import Image

class Pattern:

    def __init__(self, path):
        ref = Image.open(path)
        ref.thumbnail((40, 40), Image.LANCZOS)
        ref = np.array(ref) / 255
        ref = np.pad(ref, ((30, 30), (30, 30), (0, 0)))
        ref = np.transpose(ref, (2, 1, 0))
        self.data = ref

    def shape(self):
        return self.data.shape

    def as_image(self):
        return np.transpose(self.data, (2, 1, 0))
