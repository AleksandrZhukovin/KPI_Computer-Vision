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
        self.bin = None

    def img_to_array(self):
        img = Image.open(self.path)
        self.rgb = np.asarray(img)

    def rgb_to_greyscale(self):
        self.greyscale = np.sum(self.rgb, axis=2) / 3

    def display(self, mode):
        if mode == 'grey':
            plt.imshow(self.greyscale, cmap='grey')
            plt.show()
        elif mode == 'rgb':
            plt.imshow(self.rgb)
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

    """ Спроба реалізувати метод Крістіана """
    def cristian(self):
        shape = self.greyscale.shape
        sliced_ar = np.vstack((np.zeros((shape[1],)), self.greyscale, np.zeros((shape[1],))))
        sliced_ar = np.hstack((np.zeros((sliced_ar.shape[0], 1)), sliced_ar, np.zeros((sliced_ar.shape[0], 1))))
        windows = np.lib.stride_tricks.sliding_window_view(sliced_ar, (3, 3))
        binary = np.zeros(shape, dtype=int)
        max_std = 0
        row = 0
        min_bright = np.min(self.greyscale)
        for i in range(shape[0]):
            for w in windows[row]:
                std = np.std(w)
                if std > max_std:
                    max_std = std
            row += 1
        row = 0
        for i in range(shape[0]):
            col = 0
            for w in windows[row]:
                std = np.std(w)
                var = np.var(w)
                threshold = (1 - 0.5) * var + 0.5 * min_bright + 0.5 * (std / max_std) * (var - min_bright)
                binary[row, col] = 0 if self.greyscale[row, col] >= threshold else 1
                col += 1
            row += 1
        self.bin = binary
        # plt.imshow(binary, cmap='grey')
        # plt.show()

    """ Спроба реалізувати метод Ніблека"""
    def nibl(self, ob_color):
        shape = self.greyscale.shape
        sliced_ar = np.vstack((np.zeros((shape[1],)), self.greyscale, np.zeros((shape[1],))))
        sliced_ar = np.hstack((np.zeros((sliced_ar.shape[0], 1)), sliced_ar, np.zeros((sliced_ar.shape[0], 1))))
        windows = np.lib.stride_tricks.sliding_window_view(sliced_ar, (3, 3))
        binary = np.zeros(shape,  dtype=int)
        row = 0
        for i in range(shape[0]):
            col = 0
            for w in windows[row]:
                mid_br = (1 / 9) * np.sum(w)
                std = np.std(w)
                if ob_color == 'white':
                    threshold = mid_br + 0.2 * std
                    binary[row, col] = 1 if self.greyscale[row, col] >= threshold else 0
                elif ob_color == 'black':
                    threshold = mid_br - 0.2 * std
                    binary[row, col] = 1 if self.greyscale[row, col] >= threshold else 0
                col += 1
            row += 1
        self.bin = binary
        # plt.imshow(binary, cmap='grey')
        # plt.show()

    def cut(self):
        obj = np.multiply(np.stack((self.bin, self.bin, self.bin), axis=2), self.rgb, dtype=int, casting='unsafe')
        plt.imshow(obj)
        plt.show()


cup = Convert('images/cup.jpg')
cup.img_to_array()
cup.rgb_to_greyscale()
cup.binarization('white')
cup.cut()


watch = Convert('images/watch.jpg')
watch.img_to_array()
watch.rgb_to_greyscale()
watch.binarization('black')
watch.cut()


holmes = Convert('images/holmes.jpg')
holmes.img_to_array()
holmes.rgb_to_greyscale()
holmes.cristian()
holmes.cut()


math = Convert('images/math.jpg')
math.img_to_array()
math.rgb_to_greyscale()
math.cristian()
math.cut()
