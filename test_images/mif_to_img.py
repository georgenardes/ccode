import cv2 as cv
import os
import numpy as np
from matplotlib import pyplot as plt

WIDTH=8
DEPTH=4096
imw = 24
imh = 32
img = np.ndarray((32, 24), np.int8)


with open("../o_buff_0.mem", "r") as mem:

    print(mem.readline())
    print(mem.readline())
    print(mem.readline())

    for i in range(imh):
        for j in range(imw):
            line = mem.readline().replace("\n", "")
            val = line.split(": ")[1]
            int_val = int(val)

            img[i, j] = int_val # / 255

    print(img)

    plt.imshow(img, 'gray')
    plt.show()

with open("../o_buff_1.mem", "r") as mem:

    print(mem.readline())
    print(mem.readline())
    print(mem.readline())

    for i in range(imh):
        for j in range(imw):
            line = mem.readline().replace("\n", "")
            val = line.split(": ")[1]
            int_val = int(val)

            img[i, j] = int_val # / 255

    print(img)

    plt.imshow(img, 'gray')
    plt.show()


with open("../o_buff_2.mem", "r") as mem:

    print(mem.readline())
    print(mem.readline())
    print(mem.readline())

    for i in range(imh):
        for j in range(imw):
            line = mem.readline().replace("\n", "")
            val = line.split(": ")[1]
            int_val = int(val)

            img[i, j] = int_val # / 255

    print(img)

    plt.imshow(img, 'gray')
    plt.show()

with open("../o_buff_3.mem", "r") as mem:

    print(mem.readline())
    print(mem.readline())
    print(mem.readline())

    for i in range(imh):
        for j in range(imw):
            line = mem.readline().replace("\n", "")
            val = line.split(": ")[1]
            int_val = int(val)

            img[i, j] = int_val # / 255

    print(img)

    plt.imshow(img, 'gray')
    plt.show()


with open("../o_buff_4.mem", "r") as mem:

    print(mem.readline())
    print(mem.readline())
    print(mem.readline())

    for i in range(imh):
        for j in range(imw):
            line = mem.readline().replace("\n", "")
            val = line.split(": ")[1]
            int_val = int(val)

            img[i, j] = int_val # / 255

    print(img)

    plt.imshow(img, 'gray')
    plt.show()


with open("../o_buff_5.mem", "r") as mem:

    print(mem.readline())
    print(mem.readline())
    print(mem.readline())

    for i in range(imh):
        for j in range(imw):
            line = mem.readline().replace("\n", "")
            val = line.split(": ")[1]
            int_val = int(val)

            img[i, j] = int_val # / 255

    print(img)

    plt.imshow(img, 'gray')
    plt.show()