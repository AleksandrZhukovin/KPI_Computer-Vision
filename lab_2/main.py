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

    def scale(self, pic, x, y):
        self.scaled = pic
        for _ in range(y):
            self.scaled = np.hstack((np.zeros((self.greyscale.shape[0], 1)), self.scaled,
                                     np.zeros((self.greyscale.shape[0], 1))))
        for _ in range(x):
            self.scaled = np.vstack((np.zeros(self.scaled.shape[1]), self.scaled, np.zeros(self.scaled.shape[1])))

    def slide(self):
        self.scale(self.greyscale, 10, 20)
        core_right = np.zeros((21, 41))
        core_down = np.zeros((21, 41))
        core_right[9, 40] = 1
        core_down[20, 19] = 1
        out = np.zeros(self.greyscale.shape)
        for row in range(self.greyscale.shape[0] - 1):
            for col in range(self.greyscale.shape[1] - 1):
                out[row, col] = np.sum(np.multiply(core_right, self.scaled[row:row+21, col:col+41]))
        self.scale(out, 10, 20)
        for row in range(self.greyscale.shape[0] - 1):
            for col in range(self.greyscale.shape[1] - 1):
                out[row, col] = np.sum(np.multiply(core_down, self.scaled[row:row+21, col:col+41]))
        rescaled = out.astype(np.uint8)
        image = Image.fromarray(rescaled, mode="L")
        image.save('images/slide.jpeg')
        return out

    def gaussian(self):
        self.scale(self.greyscale, 14, 14)
        core = np.zeros((15, 15))
        a = np.linspace(2, 14, 8)
        core[7, :] = np.append(a, a[:-1][::-1])
        out = np.zeros(self.greyscale.shape)
        for row in range(6):
            line = np.linspace(2, a[row], 8)
            core[row, :] = np.append(line, line[:-1][::-1])
            core[-(row + 1), :] = np.append(line, line[:-1][::-1])
        for row in range(self.greyscale.shape[0] - 1):
            for col in range(self.greyscale.shape[1] - 1):
                out[row, col] = np.sum(np.multiply(core, self.scaled[row:row+15, col:col+15]))
        rescaled = out.astype(np.uint8)
        image = Image.fromarray(rescaled, mode="L")
        image.save('images/gaussian.jpeg')
        return out

    def inversion(self):
        self.scale(self.greyscale, 20, 20)
        core = np.zeros((41, 41))
        core[19, 19] = -1
        out = np.zeros(self.greyscale.shape)
        for row in range(self.greyscale.shape[0] - 1):
            for col in range(self.greyscale.shape[1] - 1):
                out[row, col] = np.sum(np.multiply(core, self.scaled[row:row+41, col:col+41]))
        rescaled = out.astype(np.uint8)
        image = Image.fromarray(rescaled, mode="L")
        image.save('images/inversion.jpeg')
        return out

    def move_diag(self):
        self.scale(self.greyscale, 4, 4)
        core = np.eye(9, 9)
        out = np.zeros(self.greyscale.shape)
        for row in range(self.greyscale.shape[0] - 1):
            for col in range(self.greyscale.shape[1] - 1):
                out[row, col] = np.sum(np.multiply(core, self.scaled[row:row + 9, col:col + 9]))
        rescaled = out.astype(np.uint8)
        image = Image.fromarray(rescaled, mode="L")
        image.save('images/diag_move.jpeg')
        return out

    def sharpness(self):
        self.scale(self.greyscale, 4, 4)
        core = np.zeros((9, 9))
        core[4, :] = [-1 for _ in range(9)]
        core[:, 4] = [-1 for _ in range(9)]
        core[4, 4] = 20
        out = np.zeros(self.greyscale.shape)
        for row in range(self.greyscale.shape[0] - 1):
            for col in range(self.greyscale.shape[1] - 1):
                out[row, col] = np.sum(np.multiply(core, self.scaled[row:row + 9, col:col + 9]))
        rescaled = out.astype(np.uint8)
        image = Image.fromarray(rescaled, mode="L")
        image.save('images/sharped.jpeg')
        return out

    def sobel(self):
        self.rgb_to_greyscale()
        self.scale(self.greyscale, 1, 1)
        core = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        out = np.zeros(self.greyscale.shape)
        for row in range(self.greyscale.shape[0] - 1):
            for col in range(self.greyscale.shape[1] - 1):
                out[row, col] = np.sum(np.multiply(core, self.scaled[row:row + 3, col:col + 3]))
        rescaled = out.astype(np.uint8)
        image = Image.fromarray(rescaled, mode="L")
        image.save('images/sobel_hor.jpeg')
        return out

    def borders(self):
        self.scale(self.greyscale, 1, 1)
        core = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        out = np.zeros(self.greyscale.shape)
        for row in range(self.greyscale.shape[0] - 1):
            for col in range(self.greyscale.shape[1] - 1):
                out[row, col] = np.sum(np.multiply(core, self.scaled[row:row + 3, col:col + 3]))
        rescaled = out.astype(np.uint8)
        image = Image.fromarray(rescaled, mode="L")
        image.save('images/borders.jpeg')
        return out

    def laplace(self):
        self.scale(self.greyscale, 1, 1)
        core = np.array([[0, 1, 0], [1, -8, 1], [0, 1, 0]])
        out = np.zeros(self.greyscale.shape)
        for row in range(self.greyscale.shape[0] - 1):
            for col in range(self.greyscale.shape[1] - 1):
                out[row, col] = np.sum(np.multiply(core, self.scaled[row:row + 3, col:col + 3]))
        rescaled = out.astype(np.uint8)
        image = Image.fromarray(rescaled, mode="L")
        image.save('images/laplace.jpeg')
        return out


i = Filter('images/input.png')
i.img_to_array()
i.rgb_to_greyscale()
i.display('custom', i.sobel())
