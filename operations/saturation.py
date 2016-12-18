import cv2
import numpy as np

def saturation(img, fac):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_hsv[:,:,1] += fac
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
