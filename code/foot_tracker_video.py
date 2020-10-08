#!/usr/bin/env python3
"""
Author: Daniel Ambrose
Created: 2020-09-15 16:13:33
Description: Tracks foot location of players, plots and renders video
"""
import json
import sys
import glob
import cv2
import numpy as np
from frame_draw import frame_draw
from centroidtracker import CentroidTracker

CT = CentroidTracker()

# collects key point files and puts them in to an ordered list
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

PATHOUT = "results/Center-Point_Tracking/location-point_tracking_2.avi"

# Read video
video = cv2.VideoCapture("data/match_1_rally_2_1080_60fps.mp4")

# Exit if video not opened.
if not video.isOpened():
    print( "Could not open video" )
    sys.exit()

FRAME_ARRAY = []
N = 0
while True:
    # Read a new frame
    OK, FRAME = video.read()
    if not OK:
        break
    CENTROIDS = []
    # loop over both players
    for x in CLIP_DATA[N][0].values():
        # returning centroid coordinates
        centroid = (x[3][0], x[3][1] + (round(max(x[5][1])) - x[3][1]))
        CENTROIDS.append(centroid)

    OBJECTS = CT.update(CENTROIDS)
    # loop over the tracked objects
    for (objectID, centroid) in OBJECTS.items():
        print(objectID)
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(FRAME, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.circle(FRAME, (centroid[0], centroid[1]), 8, (0, 255, 0), -1)

    #inserting the frames into an image array
    FRAME_ARRAY.append(FRAME)
    N += 1
    height, width, layers = FRAME.shape
    size = (width,height)
OUT = cv2.VideoWriter(PATHOUT,cv2.VideoWriter_fourcc(*'DIVX'), 60, size)

for i, value in enumerate(FRAME_ARRAY):
    # writing to a image array
    OUT.write(FRAME_ARRAY[i])

    # Release everything if job is finished
OUT.release()
