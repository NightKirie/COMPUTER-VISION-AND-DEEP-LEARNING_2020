import numpy as np
import cv2
from matplotlib import pyplot as plt


def disparityMap():
    imgL = cv2.imread('Q3_Image/imL.png')
    imgR = cv2.imread('Q3_Image/imR.png')
    gray_imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    gray_imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(gray_imgL, gray_imgR)
    plt.imshow(disparity, 'gray')
    plt.show()