import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import multiprocessing
from multiprocessing import Process, Queue, Pool


imgs_path = './imgs'
imgs2_path = "./imgs2"


def main():
    dataset_luminance = []
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cpu_count)
    listing = os.listdir(imgs_path)
    luminance = pool.map(process_imgs, listing)
    dataset_luminance.append(luminance)
    plot_histogram(dataset_luminance)
    pool.close()
    pool.join()


def process_imgs(i):
    input_path = os.path.join(imgs_path, i)
    img_value = cv2.imread(input_path, cv2.IMREAD_COLOR)
    luminance = calculate_luminance(img_value)
    print(luminance)
    return luminance


def plot_histogram(dataset):
    # plot histogram of luminance
    # density=False would make counts
    q25, q75 = np.percentile(dataset, [.25, .75])
    bin_width = 2*(q75 - q25)*len(dataset)**(-1/3)
    bins = round((dataset.max() - dataset.min())/bin_width)
    plt.hist(dataset, density=True, bins=bins)
    plt.ylabel('Probability')
    plt.xlabel('Data')
    plt.show()


def get_image_properties(img):
    luminance = calculate_luminance(img)
    print(luminance)
    return luminance


def calculate_luminance(img):
    luminance = []
    for x in range(len(img)):
        for y in range(len(img[x])):
            (b, g, r) = img[x, y]
            # print(
            #    "Pixel at ({}, {}) - Red: {}, Green: {}, Blue: {}".format(x, y, r, g, b))
            vR = r / 255.0
            vG = g / 255.0
            vB = b / 255.0
            Y_luminance = (0.2126 * sRGBtoLin(vR) + 0.7152 *
                           sRGBtoLin(vG) + 0.0722 * sRGBtoLin(vB))
            luminance.append(Y_luminance)

    return sum(luminance) / len(luminance)


def sRGBtoLin(colorChannel):
    # input: decimal sRGB gamma encoded color value between 0.0 and 1.0
    # output: returns a linearized value
    if(colorChannel <= 0.04045):
        return colorChannel / 12.92
    else:
        return pow(((colorChannel + 0.055)/1.055), 2.4)


if __name__ == '__main__':
    main()
