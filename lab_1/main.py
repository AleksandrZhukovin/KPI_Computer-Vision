from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import sys


np.set_printoptions(threshold=sys.maxsize)


class Convert:
    def __init__(self, path):
        self.path = path
        self.rgb = None
        self.greyscale = None
        self.mask = None

    def img_to_array(self):
        img = Image.open(self.path)
        self.rgb = np.asarray(img)

    def rgb_to_greyscale(self):
        self.greyscale = np.sum(self.rgb, axis=3) / 3

    def display(self, mode):
        if mode == 'grey':
            plt.imshow(self.greyscale, cmap='grey')
            plt.show()
        elif mode == 'rgb':
            plt.imshow(self.rgb)
            plt.show()
