#%%
import json
import itertools
import numpy as np
import cv2
import time
import glob
import os
import sys
from frame_draw import frame_draw

# n = "001"
# kp_data = "results/mamatch_1_rally_1_1080_60fps_BODY_25_MaxPeople.json/match_1_rally_1_1080_60fps_000000000{0}_keypoints.json".format(n)
# frame = "data/frames/match_1_rally_1_1080_60fps/image-{0}.jpeg".format(n)
# frame = cv2.imread(frame)

# collects key point files files and puts them in to an ordered list
files = []
for filepath in glob.glob('results/mamatch_1_rally_1_1080_60fps_BODY_25_MaxPeople.json/*.json'):
    files.append(filepath)
files = sorted(files)
# enumerates throught the key points calculating metrics related to bounding boxes and keypoint centroids
# returns a list of dicts in the form [{0:[box_origin, width, height, centroid, box_colour]}, {1: [...]}]
# where dict entries '0' and '1' are players 1 and 2 repectively
clip_data = []
for i, key_points in enumerate(files):
    with open(key_points) as json_file:
        data = json.load(json_file)
    frame_data = frame_draw(data)        
    clip_data.append(frame_data)

start = time.time()

pathOut = "results/Center-Point_Tracking/bbox_openpose.avi"

# Read video
video = cv2.VideoCapture("data/match_1_rally_1_1080_60fps.mp4")

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

    for x in clip_data[n][0].values():
        p1 = (x[0][0], x[0][1])
        p2 = (x[0][0]+x[1] , x[0][1]+x[2])
        cv2.rectangle(frame, p1, p2, x[4], 2, 1)
        cv2.circle(frame, x[3], radius=5, color=(255, 255, 255), thickness=-1)
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
# %%
