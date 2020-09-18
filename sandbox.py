#!/usr/bin/env python3
"""
Author: Daniel Ambrose
Created: 2020-09-16 16:26:36
Description: This is a sandbox file to test parts of code in development
"""
import json
import glob
import cv2
import numpy as np
from frame_draw import frame_draw
from centroidtracker import CentroidTracker

X = 801
Y = 904

CT = CentroidTracker()

# Read source image.
# Four corners of the book in source image
PTS_SRC = np.array([[600, 529], [1319, 527], [1255, 852], [672, 855]])

# Read destination image.
# Four corners of the book in destination image.
PTS_DST = np.array([[16, 10], [510, 10], [338, 559], [143, 559]])

# Calculate Homography
H_MAT, status = cv2.findHomography(PTS_SRC, PTS_DST)

FILES = []
KP_PATH = 'results/OpenPose/Key_Points/mamatch_1_rally_2_1080_60fps_BODY_25_MaxPeople.json/*.json'
for filepath in glob.glob(KP_PATH):
    FILES.append(filepath)
FILES = sorted(FILES)
# enumerates throught the key points calculating metrics related to bounding
# boxes and keypoint centroids returns a list of dicts in the form
# [{0:[box_origin, width, height, centroid, box_colour]}, {1: [...]}] where dict
# entries '0' and '1' are players 1 and 2 repectively
CLIP_DATA = []
for i, key_points in enumerate(FILES):
    with open(key_points) as json_file:
        data = json.load(json_file)
    frame_data = frame_draw(data)
    CLIP_DATA.append(frame_data)

N = 0
FRAME_ARRAY = []
CENTROIDS = []

centroid = (CLIP_DATA[0][0]['0'][3][0], CLIP_DATA[0][0]['0'][3][1] +
            (round(max(CLIP_DATA[0][0]['0'][5][1])) -
             CLIP_DATA[0][0]['0'][3][1]))
CENTROIDS.append(centroid)
def translate_point(x,y,h):
    denom = h[2,0] *x + h[2,1] * y + h[2,2]
    xPrime = (h[0,0] *x + h[0,1] * y + h[0,2])/ denom
    yPrime = (h[1,0] *x + h[1,1] * y + h[1,2]) / denom
    return int(xPrime), int(yPrime)
X_PRIME, Y_PRIME = translate_point(CENTROIDS[0][0], CENTROIDS[0][1], H_MAT)
print(X_PRIME, Y_PRIME)

frame = cv2.imread("court_plot.png")

text = "ID 1"
cv2.putText(frame, text, (X_PRIME - 10, Y_PRIME - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.circle(frame, (X_PRIME, Y_PRIME), 8, (0, 255, 0), -1)

cv2.imwrite("sandbox.jpg", frame)
