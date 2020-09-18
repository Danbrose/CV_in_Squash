#!/usr/bin/env python3
"""
Author: Daniel Ambrose
Created: 2020-09-15 15:30:07
Description: Homography calculation module 
"""
import cv2
import numpy as np

# Read source image.
im_src = cv2.imread('sample.jpg')
# Four corners of the book in source image
pts_src = np.array([[600, 529], [1319, 527], [1255, 852], [672, 855]])

# Read destination image.
im_dst = cv2.imread('court_plot.png')
# Four corners of the book in destination image.
pts_dst = np.array([[16, 10],[510, 10], [338, 559], [143, 559]])

# Calculate Homography
h, status = cv2.findHomography(pts_src, pts_dst)
print(h)  
print(status) 
# Warp source image to destination based on homography
print(im_src)
im_out = cv2.warpPerspective(im_src, h, (520, 775))
print(im_dst.shape[1], im_dst.shape[0])
# Display images
cv2.imwrite("results/Homography/Source Image.jpg", im_src)
cv2.imwrite("results/Homography/Destination Image.jpg", im_dst)
cv2.imwrite("results/Homography/Warped Source Image.jpg", im_out)
