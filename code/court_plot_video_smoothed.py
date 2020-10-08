#!/usr/bin/env python3
"""
Author: Daniel Ambrose
Created: 2020-09-18 09:02:19
Description: Smooths tracking points and plots on court plot
"""
import json
import glob
import cv2
import numpy as np
from scipy.signal import savgol_filter
from frame_draw import frame_draw
from centroidtracker import CentroidTracker

# Intialise centroid tracker class
CT = CentroidTracker()

# Four corners of the court in frame
PTS_SRC = np.array([[600, 529], [1319, 527], [1560, 852], [365, 855]])
# Four corners of the court in plot
PTS_DST = np.array([[16, 10], [510, 10], [510, 559], [16, 559]])
H_MAT, _ = cv2.findHomography(PTS_SRC, PTS_DST)

# collects key point FILES and puts them in to an ordered list
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

PATHOUT = "results/Court_Plots/court_plot_smoothed_2.avi"

def translate_point(x,y,h):
    denom = h[2,0] *x + h[2,1] * y + h[2,2]
    xPrime = (h[0,0] *x + h[0,1] * y + h[0,2])/ denom
    yPrime = (h[1,0] *x + h[1,1] * y + h[1,2]) / denom
    return np.array([int(xPrime), int(yPrime)])

FRAME_ARRAY = []
TRACK_POINTS = {
    "0": {
        "X": [],
        "Y": []
    },
    "1": {
        "X": [],
        "Y": []
    },
}

# Loop over all frames
for f in CLIP_DATA:
    # loop over both players
    CENTROIDS = []
    for x in f[0].values():
        centroid = (x[3][0], x[3][1] + (round(max(x[5][1])) - x[3][1]))
        CENTROIDS.append(centroid)

    objects = CT.update(CENTROIDS)
    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        centroid = translate_point(centroid[0], centroid[1], H_MAT)
        TRACK_POINTS['{}'.format(objectID)]["X"].append(centroid[0])
        TRACK_POINTS['{}'.format(objectID)]["Y"].append(centroid[1])

INTERVAL = 31
POLY = 2
TRACK_POINTS['0']['X'] = savgol_filter(TRACK_POINTS['0']['X'], INTERVAL, POLY).round().astype(int)
TRACK_POINTS['0']['Y'] = savgol_filter(TRACK_POINTS['0']['Y'], INTERVAL, POLY).round().astype(int)
TRACK_POINTS['1']['X'] = savgol_filter(TRACK_POINTS['1']['X'], INTERVAL, POLY).round().astype(int)
TRACK_POINTS['1']['Y'] = savgol_filter(TRACK_POINTS['1']['Y'], INTERVAL, POLY).round().astype(int)

PLAYER_0 = []
PLAYER_1 = []
for (PLAYER_ID, COORDINATES) in TRACK_POINTS.items():
    for X, Y in zip(COORDINATES['X'], COORDINATES['Y']):
        if PLAYER_ID == '0':
            PLAYER_0.append([PLAYER_ID, X, Y])
        else:
            PLAYER_1.append([PLAYER_ID, X, Y])

for FIRST, SECOND in zip(PLAYER_0, PLAYER_1):
    frame = cv2.imread("court_plot.png")
    # draw both the ID of the object and the centroid of the
    # object on the output frame
    text = "ID {}".format(FIRST[0])
    cv2.putText(frame, text, (FIRST[1] - 10, FIRST[2] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.circle(frame, (FIRST[1], FIRST[2]), 8, (0, 255, 0), -1)

    text = "ID {}".format(SECOND[0])
    cv2.putText(frame, text, (SECOND[1] - 10, SECOND[2] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.circle(frame, (SECOND[1], SECOND[2]), 8, (0, 255, 0), -1)

    #inserting the frames into an image array
    FRAME_ARRAY.append(frame)

frame = cv2.imread("court_plot.png")
height, width, layers = frame.shape
size = (width, height)

OUT = cv2.VideoWriter(PATHOUT, cv2.VideoWriter_fourcc(*'DIVX'), 60, size)

for i, value in enumerate(FRAME_ARRAY):
    # writing to a image array
    OUT.write(value)

    # Release everything if job is finished
OUT.release()
