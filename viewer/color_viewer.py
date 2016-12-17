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

from util_planck import *
#from get_patch import *


if len(sys.argv) < 2:
    print "Usage: ./colorviewer.py input_image"
    sys.exit(0)
f = sys.argv[1]

# function to give a list of points on the planckian blackbody locus
planck_points_cie_2 = calc_planck_locus_single_poly()

img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)

step=5
colors_flat = []
for x in range(0,img.shape[1],step):
    for y in range(0,img.shape[0],step):
        colors_flat.append(img[y,x])
colors_flat = np.array(colors_flat)

######################################### figure
fig = plt.figure(figsize=(20,10))
gs = gridspec.GridSpec(2, 3, width_ratios=[1,1,1], height_ratios=[1,1]) # FIXME: was height_rations=[1,1,1]

ax = plt.subplot(gs[0])
ax.imshow(img) # show original image with the selected box
ax.set_title('Image')

ax = plt.subplot(gs[1], projection='3d')
for p in colors_flat:
    ax.scatter(p[0], p[1], p[2], '.', c=[p[0]/255.,p[1]/255.,p[2]/255.])

ax.set_xlim([0, 255])
ax.set_ylim([0, 255])
ax.set_zlim([0, 255])

ax.set_xlabel('red')
ax.set_ylabel('green')
ax.set_zlabel('blue')
ax.set_title('RGB values')

ax = plt.subplot(gs[2])
CIE_1931_chromaticity_diagram_plot(standalone=False)

# plot the points of the planckian blackbody locus
plt.plot(planck_points_cie_2[:,0], planck_points_cie_2[:,1], 'k-')
for c in colors_flat:
    rgb = sRGBColor(c[0]/255.,c[1]/255.,c[2]/255.)
    xyY = convert_color(rgb, xyYColor)
    plt.scatter(xyY.xyy_x, xyY.xyy_y)


ax = plt.subplot(gs[3])
plt.hist(colors_flat, 256, color=['red','green','blue'], histtype='step')
plt.axvline(np.mean(colors_flat[:,0]), color='red')
plt.axvline(np.mean(colors_flat[:,1]), color='green')
plt.axvline(np.mean(colors_flat[:,2]), color='blue')
plt.title("histog")

ax = plt.subplot(gs[4])

ax = plt.subplot(gs[5])


plt.show()
