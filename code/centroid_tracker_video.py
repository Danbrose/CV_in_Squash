#!/usr/bin/env python3
from centroidtracker import CentroidTracker
import json
import itertools
import numpy as np
import cv2
import time
import glob
import os
import sys
from frame_draw import frame_draw

ct = CentroidTracker()

# collects key point files files and puts them in to an ordered list
files = []
for filepath in glob.glob('results/mamatch_1_rally_2_1080_60fps_BODY_25_MaxPeople.json/*.json'):
    files.append(filepath)
files = sorted(files)
# enumerates throught the key points calculating metrics related to bounding
# boxes and keypoint centroids returns a list of dicts in the form
# [{0:[box_origin, width, height, centroid, box_colour]}, {1: [...]}] where dict
# entries '0' and '1' are players 1 and 2 repectively
clip_data = []
for i, key_points in enumerate(files):
    with open(key_points) as json_file:
        data = json.load(json_file)
    frame_data = frame_draw(data)        
    clip_data.append(frame_data)

start = time.time()

pathOut = "center-point_tracking_2.avi"

# Read video
video = cv2.VideoCapture("match_1_rally_2_1080_60fps.mp4")

# Exit if video not opened.
if not video.isOpened():
    print( "Could not open video" )
    sys.exit()

frame_array = []
n = 0
while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break
    rects = []
    centroids = []
    # loop over both players
    for x in clip_data[n][0].values():
        # Ploting bounding box from XY start and ends
        # p1 = (x[0][0], x[0][1])
        # p2 = (x[0][0]+x[1] , x[0][1]+x[2])
        # cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
        # box = (x[0][0], x[0][1], x[0][0]+x[1] , x[0][1]+x[2])
        # rects.append(box)
        # returning centroid coordinates
        centroid = x[3]
        centroids.append(centroid)

    objects = ct.update(centroids)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 8, (0, 255, 0), -1)

    #inserting the frames into an image array
    frame_array.append(frame)
    n += 1
    height, width, layers = frame.shape
    size = (width,height)
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), 60, size)

for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])

    # Release everything if job is finished
out.release()
end = time.time()
duration = end - start
