# GitHub: https://github.com/AleksandrZhukovin/KPI_Computer-Vision

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
from noise import sp_noise_color


class Filter:
    def __init__(self, path):
        self.path = path
        self.rgb = None
        self.hsv = None
        self.greyscale = None
        self.scaled = None
        self.norm_noise = None
        self.sp_noise = None

    def img_to_array(self):
        img = np.array(Image.open(self.path))
        img.setflags(write=1)
        self.rgb = np.asarray(img)

    def rgb_to_greyscale(self):
        self.greyscale = np.sum(self.rgb, axis=2) / 3

    def hsv_open(self):
        self.hsv = cv2.imread(self.path, cv2.COLOR_BGR2HSV)

    def display(self, mode, pic=None, cmap=None):
        if mode == 'grey':
            plt.imshow(self.greyscale, cmap='grey')
            plt.show()
            plt.imshow(self.scaled, cmap='grey')
            plt.show()
        elif mode == 'rgb':
            plt.imshow(self.rgb)
            plt.show()
        elif mode == 'custom':
            if cmap == 'grey':
                plt.imshow(pic, cmap='grey')
            else:
                plt.imshow(pic)
            plt.show()
        elif mode == 'hsv':
            cv2.imshow('HSV image', self.hsv)
            cv2.waitKey(0)

    def noise(self, name):
        gauss_noise = np.zeros(self.rgb.shape, dtype=np.uint8)
        cv2.randn(gauss_noise, 0, 20)
        gauss_noise = (gauss_noise * 10).astype(np.uint8)
        self.norm_noise = cv2.add(self.rgb, gauss_noise)
        cv2.imwrite(f'images/normal_noisy_{name}.jpg', self.norm_noise)

        self.sp_noise = sp_noise_color(self.rgb)
        cv2.imwrite(f'images/sp_noisy_{name}.jpg', self.sp_noise)

    def noise_hsv(self, name):
        gauss_noise = np.zeros(self.hsv.shape, dtype=np.uint8)
        cv2.randn(gauss_noise, 0, 20)
        gauss_noise = (gauss_noise * 10).astype(np.uint8)
        self.norm_noise = cv2.add(self.hsv, gauss_noise)
        cv2.imwrite(f'images/HSV_normal_noisy_{name}.jpg', self.norm_noise)

        self.sp_noise = sp_noise_color(self.hsv)
        cv2.imwrite(f'images/HSV_sp_noisy_{name}.jpg', self.sp_noise)

    def box_average(self, noise, name):
        out = np.zeros(self.rgb.shape)
        if noise == 'normal':
            for row in range(self.rgb.shape[0] - 5):
                for col in range(self.rgb.shape[1] - 5):
                    out[row, col, 0] = np.sum(self.norm_noise[row:row+5, col:col+5, 0]) / 25
                    out[row, col, 1] = np.sum(self.norm_noise[row:row + 5, col:col + 5, 1]) / 25
                    out[row, col, 2] = np.sum(self.norm_noise[row:row + 5, col:col + 5, 2]) / 25
            cv2.imwrite(f'images/normal_filtered_average_{name}.jpg', out)
        elif noise == 'sp':
            for row in range(self.rgb.shape[0] - 5):
                for col in range(self.rgb.shape[1] - 5):
                    out[row, col, 0] = np.sum(self.sp_noise[row:row + 5, col:col + 5, 0]) / 25
                    out[row, col, 1] = np.sum(self.sp_noise[row:row + 5, col:col + 5, 1]) / 25
                    out[row, col, 2] = np.sum(self.sp_noise[row:row + 5, col:col + 5, 2]) / 25
            cv2.imwrite(f'images/sp_filtered_average_{name}.jpg', out)
        return out

    def box_average_hsv(self, noise, name):
        out = np.zeros(self.hsv.shape)
        if noise == 'normal':
            for row in range(self.hsv.shape[0] - 5):
                for col in range(self.hsv.shape[1] - 5):
                    out[row, col, 0] = np.sum(self.norm_noise[row:row+5, col:col+5, 0]) / 25
                    out[row, col, 1] = np.sum(self.norm_noise[row:row + 5, col:col + 5, 1]) / 25
                    out[row, col, 2] = np.sum(self.norm_noise[row:row + 5, col:col + 5, 2]) / 25
            cv2.imwrite(f'images/HSV_normal_filtered_average_{name}.jpg', out)
        elif noise == 'sp':
            for row in range(self.hsv.shape[0] - 5):
                for col in range(self.hsv.shape[1] - 5):
                    out[row, col, 0] = np.sum(self.sp_noise[row:row + 5, col:col + 5, 0]) / 25
                    out[row, col, 1] = np.sum(self.sp_noise[row:row + 5, col:col + 5, 1]) / 25
                    out[row, col, 2] = np.sum(self.sp_noise[row:row + 5, col:col + 5, 2]) / 25
            cv2.imwrite(f'images/HSV_sp_filtered_average_{name}.jpg', out)
        return out

    def median(self, noise, name):
        out = np.zeros(self.rgb.shape)
        if noise == 'normal':
            for row in range(self.rgb.shape[0] - 5):
                for col in range(self.rgb.shape[1] - 5):
                    out[row, col, 0] = np.median(self.norm_noise[row:row+5, col:col+5, 0])
                    out[row, col, 1] = np.median(self.norm_noise[row:row + 5, col:col + 5, 1])
                    out[row, col, 2] = np.median(self.norm_noise[row:row + 5, col:col + 5, 2])
            cv2.imwrite(f'images/normal_filtered_median_{name}.jpg', out)
        elif noise == 'sp':
            for row in range(self.rgb.shape[0] - 5):
                for col in range(self.rgb.shape[1] - 5):
                    out[row, col, 0] = np.median(self.sp_noise[row:row+5, col:col+5, 0])
                    out[row, col, 1] = np.median(self.sp_noise[row:row + 5, col:col + 5, 1])
                    out[row, col, 2] = np.median(self.sp_noise[row:row + 5, col:col + 5, 2])
            cv2.imwrite(f'images/sp_filtered_median_{name}.jpg', out)
        return out

    def median_hsv(self, noise, name):
        out = np.zeros(self.hsv.shape)
        if noise == 'normal':
            for row in range(self.hsv.shape[0] - 5):
                for col in range(self.hsv.shape[1] - 5):
                    out[row, col, 0] = np.median(self.norm_noise[row:row+5, col:col+5, 0])
                    out[row, col, 1] = np.median(self.norm_noise[row:row + 5, col:col + 5, 1])
                    out[row, col, 2] = np.median(self.norm_noise[row:row + 5, col:col + 5, 2])
            cv2.imwrite(f'images/HSV_normal_filtered_median_{name}.jpg', out)
        elif noise == 'sp':
            for row in range(self.hsv.shape[0] - 5):
                for col in range(self.hsv.shape[1] - 5):
                    out[row, col, 0] = np.median(self.sp_noise[row:row+5, col:col+5, 0])
                    out[row, col, 1] = np.median(self.sp_noise[row:row + 5, col:col + 5, 1])
                    out[row, col, 2] = np.median(self.sp_noise[row:row + 5, col:col + 5, 2])
            cv2.imwrite(f'images/HSV_sp_filtered_median_{name}.jpg', out)
        return out

    def weighted_median(self, noise, name):
        core = np.ones((3, 3, 3))
        core[1, 0:, :] = [2, 2, 2]
        core[0:, 1, :] = [2, 2, 2]
        core[1, 1, :] = 4
        core = core / 9
        out = np.zeros(self.rgb.shape)
        if noise == 'normal':
            for row in range(self.rgb.shape[0] - 3):
                for col in range(self.rgb.shape[1] - 3):
                    out[row, col, 0] = np.median(np.multiply(core[:, :, 0], self.norm_noise[row:row + 3, col:col + 3, 0]))
                    out[row, col, 1] = np.median(np.multiply(core[:, :, 1], self.norm_noise[row:row + 3, col:col + 3, 1]))
                    out[row, col, 2] = np.median(np.multiply(core[:, :, 2], self.norm_noise[row:row + 3, col:col + 3, 2]))
            cv2.imwrite(f'images/normal_filtered_weight_{name}.jpg', out * 4)
        elif noise == 'sp':
            for row in range(self.rgb.shape[0] - 3):
                for col in range(self.rgb.shape[1] - 3):
                    out[row, col, 0] = np.median(np.multiply(core[:, :, 0], self.sp_noise[row:row + 3, col:col + 3, 0]))
                    out[row, col, 1] = np.median(np.multiply(core[:, :, 1], self.sp_noise[row:row + 3, col:col + 3, 1]))
                    out[row, col, 2] = np.median(np.multiply(core[:, :, 2], self.sp_noise[row:row + 3, col:col + 3, 2]))
            cv2.imwrite(f'images/sp_filtered_weight_{name}.jpg', out * 4)
        return out

    def weighted_median_hsv(self, noise, name):
        core = np.ones((3, 3, 3))
        core[1, 0:, :] = [2, 2, 2]
        core[0:, 1, :] = [2, 2, 2]
        core[1, 1, :] = 4
        core = core / 9
        out = np.zeros(self.hsv.shape)
        if noise == 'normal':
            for row in range(self.hsv.shape[0] - 3):
                for col in range(self.hsv.shape[1] - 3):
                    out[row, col, 0] = np.median(np.multiply(core[:, :, 0], self.norm_noise[row:row + 3, col:col + 3, 0]))
                    out[row, col, 1] = np.median(np.multiply(core[:, :, 1], self.norm_noise[row:row + 3, col:col + 3, 1]))
                    out[row, col, 2] = np.median(np.multiply(core[:, :, 2], self.norm_noise[row:row + 3, col:col + 3, 2]))
            cv2.imwrite(f'images/HSV_normal_filtered_weight_{name}.jpg', out * 4)
        elif noise == 'sp':
            for row in range(self.hsv.shape[0] - 3):
                for col in range(self.hsv.shape[1] - 3):
                    out[row, col, 0] = np.median(np.multiply(core[:, :, 0], self.sp_noise[row:row + 3, col:col + 3, 0]))
                    out[row, col, 1] = np.median(np.multiply(core[:, :, 1], self.sp_noise[row:row + 3, col:col + 3, 1]))
                    out[row, col, 2] = np.median(np.multiply(core[:, :, 2], self.sp_noise[row:row + 3, col:col + 3, 2]))
            cv2.imwrite(f'images/HSV_sp_filtered_weight_{name}.jpg', out * 4)
        return out


# image = Filter('images/truck.jpg')
# image.img_to_array()
# image.rgb_to_greyscale()
# image.noise('truck')
# image.display('custom', image.norm_noise)
# image.display('custom', image.sp_noise)
# image.weighted_median('sp', 'truck')
# image.weighted_median('normal', 'truck')
# image.median('sp', 'truck')
# image.median('normal', 'truck')
# image.box_average('sp', 'truck')
# image.box_average('normal', 'truck')

# image = Filter('images/starship.jpg')
# image.img_to_array()
# image.rgb_to_greyscale()
# image.noise('starship')
# image.display('custom', image.norm_noise)
# image.display('custom', image.sp_noise)
# image.weighted_median('sp', 'starship')
# image.weighted_median('normal', 'starship')
# image.median('sp', 'starship')
# image.median('normal', 'starship')
# image.box_average('sp', 'starship')
# image.box_average('normal', 'starship')


# image = Filter('images/truck.jpg')
# image.hsv_open()
# image.noise_hsv('truck')
# image.median_hsv('normal', 'truck')
# image.median_hsv('sp', 'truck')
# image.box_average_hsv('normal', 'truck')
# image.box_average_hsv('sp', 'truck')
# image.weighted_median_hsv('normal', 'truck')
# image.weighted_median_hsv('sp', 'truck')

# image = Filter('images/fighter.jpg')
# image.hsv_open()
# image.noise_hsv('fighter')
# image.median_hsv('normal', 'fighter')
# image.median_hsv('sp', 'fighter')
# image.box_average_hsv('normal', 'fighter')
# image.box_average_hsv('sp', 'fighter')
# image.weighted_median_hsv('normal', 'fighter')
# image.weighted_median_hsv('sp', 'fighter')
#
# image = Filter('images/starship.jpg')
# image.hsv_open()
# image.noise_hsv('starship')
# image.median_hsv('normal', 'starship')
# image.median_hsv('sp', 'starship')
# image.box_average_hsv('normal', 'starship')
# image.box_average_hsv('sp', 'starship')
# image.weighted_median_hsv('normal', 'starship')
# image.weighted_median_hsv('sp', 'starship')
