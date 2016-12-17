#!/usr/bin/python
'''
author: Tobi

given an input image, show some statistics about it
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt

import sys
from math import sqrt, atan2
from grapefruit import Color
import matplotlib.pyplot as plt
import numpy

import colour
from colour.plotting import *
import matplotlib
import os
import pylab

import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

from colormath.color_objects import XYZColor, sRGBColor, xyYColor, HSVColor
from colormath.color_conversions import convert_color

import sys
sys.path.append("../")

from operations.whitebalance import *

from util_planck import *
#from get_patch import *


if len(sys.argv) < 2:
    print "Usage: ./colorviewer.py input_image"
    sys.exit(0)
f = sys.argv[1]

# function to give a list of points on the planckian blackbody locus
planck_points_cie_2 = calc_planck_locus_single_poly()

def flatten_image(img,step=10):
    colors_flat = []
    for x in range(0,img.shape[1],step):
        for y in range(0,img.shape[0],step):
            colors_flat.append(img[y,x])
    colors_flat = np.array(colors_flat)

    return colors_flat

img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
img_grey_world = grey_world(img)
img_retinex = retinex(img)
img_max_white = max_white(img)

step = 10
pixels = flatten_image(img,step)
pixels_grey_world = flatten_image(img_grey_world,step)
pixels_retinex = flatten_image(img_retinex,step)
pixels_max_white = flatten_image(img_max_white,step)

######################################### figure
def plot_figure(imgs, pixelss, titles):
    fig = plt.figure(figsize=(20,10))
    cols = 4

    gs = gridspec.GridSpec(len(imgs), cols, width_ratios=[1,1,1,1], height_ratios=np.zeros(len(imgs))+1)

    cnt = 0
    for img, pixels in zip(imgs,pixelss):

        ax = plt.subplot(gs[cnt*cols+0])
        plt.title("%s: Image" % titles[cnt])
        ax.imshow(img) # show original image with the selected box

        ax = plt.subplot(gs[cnt*cols+1], projection='3d')
        plt.title("%s: Pixels in RGB" % titles[cnt])
        for p in pixels:
            ax.scatter(p[0], p[1], p[2], '.', c=[p[0]/255.,p[1]/255.,p[2]/255.])

        ax.set_xlim([0, 255])
        ax.set_ylim([0, 255])
        ax.set_zlim([0, 255])

        ax.set_xlabel('red')
        ax.set_ylabel('green')
        ax.set_zlabel('blue')

        ax.view_init(elev=25., azim=210)

        ax = plt.subplot(gs[cnt*cols+2])
        CIE_1931_chromaticity_diagram_plot(standalone=False)

        ax.set_title("%s: Pixels in CIE" % titles[cnt])

        # plot the points of the planckian blackbody locus
        plt.plot(planck_points_cie_2[:,0], planck_points_cie_2[:,1], 'k-')
        for c in pixels:
            rgb = sRGBColor(c[0]/255.,c[1]/255.,c[2]/255.)
            xyY = convert_color(rgb, xyYColor)
            plt.scatter(xyY.xyy_x, xyY.xyy_y)


        ax = plt.subplot(gs[cnt*cols+3])
        plt.title("%s: Histogram" % titles[cnt])
        plt.hist(pixels, 256, color=['red','green','blue'], histtype='step')
        plt.axvline(np.mean(pixels[:,0]), color='red')
        plt.axvline(np.mean(pixels[:,1]), color='green')
        plt.axvline(np.mean(pixels[:,2]), color='blue')

        cnt += 1

imgs = []
imgs.append(img)
imgs.append(img_grey_world)
#imgs.append(img_retinex)
#imgs.append(img_max_white)

pixelss = []
pixelss.append(pixels)
pixelss.append(pixels_grey_world)
#pixels.append(pixels_retinex)
#pixels.append(pixels_max_white)

plot_figure(imgs, pixelss, ["Input", "Grey world", "Retinex", "Max White"])

plt.show()
