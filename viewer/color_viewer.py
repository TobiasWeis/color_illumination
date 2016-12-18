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
from operations.contrast import *
from operations.saturation import *

from util_planck import *

class Exp():
    def __init__(self, img, title):
        self.img = img
        self.pixels = flatten_image(img)
        self.title = title

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
experiments = []
experiments.append(Exp(img, "Input"))
experiments.append(Exp(grey_world(img), "Grey world"))
experiments.append(Exp(retinex(img), "Retinex"))
experiments.append(Exp(contrast(img,fac=2.), "Contrast:2"))
experiments.append(Exp(contrast(img,fac=.5), "Contrast:.5"))

######################################### figure
def plot_figure(experiments):
    fig = plt.figure(figsize=(20,10))
    cols = 4

    gs = gridspec.GridSpec(len(experiments), cols, width_ratios=[1,1,1,1], height_ratios=np.zeros(len(experiments))+1)

    cnt = 0
    for exp in experiments:

        ax = plt.subplot(gs[cnt*cols+0])
        plt.title("%s: Image" % exp.title)
        ax.imshow(exp.img) # show original image with the selected box

        ax = plt.subplot(gs[cnt*cols+1], projection='3d')
        plt.title("%s: Pixels in RGB" % exp.title)
        for p in exp.pixels:
            ax.scatter(p[0], p[1], p[2], '.', c=[p[0]/255.,p[1]/255.,p[2]/255.])

        ax.set_xlim([0, 255])
        ax.set_ylim([0, 255])
        ax.set_zlim([0, 255])

        ax.set_xlabel('red')
        ax.set_ylabel('green')
        ax.set_zlabel('blue')

        ax.view_init(elev=40., azim=210)

        ax = plt.subplot(gs[cnt*cols+2])
        CIE_1931_chromaticity_diagram_plot(standalone=False)

        ax.set_title("%s: Pixels in CIE" % exp.title)

        # plot the points of the planckian blackbody locus
        plt.plot(planck_points_cie_2[:,0], planck_points_cie_2[:,1], 'k-')
        for c in exp.pixels:
            rgb = sRGBColor(c[0]/255.,c[1]/255.,c[2]/255.)
            xyY = convert_color(rgb, xyYColor)
            plt.scatter(xyY.xyy_x, xyY.xyy_y)


        ax = plt.subplot(gs[cnt*cols+3])
        plt.title("%s: Histogram" % exp.title)
        plt.hist(exp.pixels, 256, color=['red','green','blue'], histtype='step')
        plt.axvline(np.mean(exp.pixels[:,0]), color='red')
        plt.axvline(np.mean(exp.pixels[:,1]), color='green')
        plt.axvline(np.mean(exp.pixels[:,2]), color='blue')

        cnt += 1


plot_figure(experiments)

plt.tight_layout()
plt.show()
