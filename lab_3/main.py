# GitHub: https://github.com/AleksandrZhukovin/KPI_Computer-Vision

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


class Filter:
    def __init__(self, path):
        self.path = path
        self.rgb = None
        self.greyscale = None
        self.scaled = None
        self.bin = None

    def img_to_array(self):
        img = Image.open(self.path)
        self.rgb = np.asarray(img)

    def rgb_to_greyscale(self):
        self.greyscale = np.sum(self.rgb, axis=2) / 3

    def display(self, mode, pic=None):
        if mode == 'grey':
            plt.imshow(self.greyscale, cmap='grey')
            plt.show()
            plt.imshow(self.scaled, cmap='grey')
            plt.show()
        elif mode == 'rgb':
            plt.imshow(self.rgb)
            plt.show()
        elif mode == 'custom':
            plt.imshow(pic, cmap='grey')
            plt.show()

    def binarization(self, ob_color):
        shape = self.greyscale.shape
        row = 0
        mean = np.mean(self.greyscale)
        binary = np.zeros(shape, dtype=int)
        for i in range(shape[0]):
            col = 0
            for w in self.greyscale[row]:
                if ob_color == 'black':
                    binary[row, col] = 1 if self.greyscale[row, col] < mean else 0
                elif ob_color == 'white':
                    binary[row, col] = 1 if self.greyscale[row, col] > mean else 0
                col += 1
            row += 1
        self.bin = binary

    def erosion(self):
        core = np.ones((5, 5))
        core[:, 1:2] = [[1 for _ in range(1, 2)] for i in range(5)]
        out = np.zeros(self.bin.shape)
        for row in range(self.bin.shape[0] - 1):
            for col in range(self.bin.shape[1] - 1):
                if np.array_equal(core, self.bin[row:row + 5, col:col + 5]):
                    out[row, col] = 1
        return out

    def dilation(self):
        core = np.ones((5, 5))
        core[:, 1:2] = [[1 for _ in range(1, 2)] for i in range(5)]
        out = np.zeros(self.bin.shape)
        for row in range(2, self.bin.shape[0] - 3):
            for col in range(2, self.bin.shape[1] - 3):
                for row1 in range(core.shape[0] - 1):
                    for col1 in range(core.shape[1] - 1):
                        if core[row1, col1] == self.bin[row:row + 5, col:col + 5][row1, col1]:
                            out[row, col] = 1
                            break
        return out

    def opening(self):
        core = np.ones((5, 5))
        out = np.zeros(self.bin.shape)
        for row in range(self.bin.shape[0] - 1):
            for col in range(self.bin.shape[1] - 1):
                if np.array_equal(core, self.bin[row:row + 5, col:col + 5]):
                    out[row, col] = 1
        for row in range(2, self.bin.shape[0] - 3):
            for col in range(2, self.bin.shape[1] - 3):
                for row1 in range(core.shape[0] - 1):
                    for col1 in range(core.shape[1] - 1):
                        if core[row1, col1] == self.bin[row:row + 5, col:col + 5][row1, col1]:
                            out[row, col] = 1
                            break
        return out

    def closing(self):
        core = np.ones((5, 5))
        out = np.zeros(self.bin.shape)
        for row in range(2, self.bin.shape[0] - 3):
            for col in range(2, self.bin.shape[1] - 3):
                for row1 in range(core.shape[0] - 1):
                    for col1 in range(core.shape[1] - 1):
                        if core[row1, col1] == self.bin[row:row + 5, col:col + 5][row1, col1]:
                            out[row, col] = 1
                            break
        for row in range(self.bin.shape[0] - 1):
            for col in range(self.bin.shape[1] - 1):
                if np.array_equal(core, self.bin[row:row + 5, col:col + 5]):
                    out[row, col] = 1
        return out

    def horizontal_border(self):
        core = np.zeros((5, 5))
        core[4, :] = [1 for i in range(5)]
        out = np.zeros(self.bin.shape)
        for row in range(self.bin.shape[0] - 1):
            for col in range(self.bin.shape[1] - 1):
                if np.array_equal(core, self.bin[row:row + 5, col:col + 5]):
                    out[row, col] = 1
        return out


f = Filter('data/holmes.jpg')
f.img_to_array()
f.rgb_to_greyscale()
f.binarization('black')
# f.display('custom', f.bin)
# f.display('custom', f.erosion())
# f.display('custom', f.dilation())
# f.display('custom', f.opening())
# f.display('custom', f.closing())
# f.display('custom', f.horizontal_border())
