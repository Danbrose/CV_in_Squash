#!/usr/bin/env python3
"""
Author: Daniel Ambrose
Created: 2020-09-22 10:04:17
Description: Unpacks annotation grounf truth data and processes
"""
import json
import glob
import sys
import cv2
import xmltodict
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import ticker as ticker
import seaborn as sns
from scipy.signal import savgol_filter
from scipy import stats, integrate
from frame_draw import frame_draw
from centroidtracker import CentroidTracker

CLIP = 'match_1_rally_1'
# CLIP = 'match_1_rally_2'

# Intialise centroid tracker class
CT = CentroidTracker()

# Four corners of the court in frame
PTS_SRC = np.array([[600, 529], [1319, 527], [1560, 852], [365, 855]])
# Four corners of the court in plot
PTS_DST = np.array([[16, 10], [510, 10], [510, 559], [16, 559]])
H_MAT, _ = cv2.findHomography(PTS_SRC, PTS_DST)

def translate_point(x,y,h):
    denom = h[2, 0] *x + h[2, 1] * y + h[2, 2]
    xPrime = (h[0, 0] *x + h[0, 1] * y + h[0, 2])/ denom
    yPrime = (h[1, 0] *x + h[1, 1] * y + h[1, 2]) / denom
    return np.array([int(xPrime), int(yPrime)])

with open('data/annotated_data/{}_annotations.xml'.format(CLIP)) as fd:
    doc = xmltodict.parse(fd.read())

# Initialise an empty list to put a dict in it to turn into a df
DATA = []
for ID in [0, 1]:
    for i, metric in enumerate(doc['annotations']['track'][ID]['points']):
        frame = metric['@frame']
        points = [int(float(x) +.5) for x in metric['@points'].split(',')]
        points = translate_point(points[0], points[1], H_MAT)
        DATA.append(
            {"frame" : frame,
             "ID" : ID,
             "GT_X" : points[0],
             "GT_Y" : points[1]}
        )
GT_df = pd.DataFrame(DATA)


# collects key point FILES and puts them in to an ordered list
FILES = []
KP_PATH = 'results/OpenPose/Key_Points/ma{}_1080_60fps_BODY_25_MaxPeople.json/*.json'.format(CLIP)
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

# Clip 1
TRACK_X = TRACK_POINTS['1']['X'] + TRACK_POINTS['0']['X']
TRACK_Y = TRACK_POINTS['1']['Y'] + TRACK_POINTS['0']['Y']

# Clip 2
# TRACK_X = TRACK_POINTS['0']['X'] + TRACK_POINTS['1']['X']
# TRACK_Y = TRACK_POINTS['0']['Y'] + TRACK_POINTS['1']['Y']

GT_df['tracking_x'] = TRACK_X
GT_df['tracking_y'] = TRACK_Y

print(GT_df)

INTERVAL = 31
POLY = 2
TRACK_POINTS['0']['X'] = savgol_filter(TRACK_POINTS['0']['X'], INTERVAL,
                                       POLY).round().astype(int)
TRACK_POINTS['0']['Y'] = savgol_filter(TRACK_POINTS['0']['Y'], INTERVAL,
                                       POLY).round().astype(int)
TRACK_POINTS['1']['X'] = savgol_filter(TRACK_POINTS['1']['X'], INTERVAL,
                                       POLY).round().astype(int)
TRACK_POINTS['1']['Y'] = savgol_filter(TRACK_POINTS['1']['Y'], INTERVAL,
                                       POLY).round().astype(int)

PIXELS2METERS = 6.4/775

# Clip 1
PLAYER_X = TRACK_POINTS['1']['X'].T
PLAYER_X = np.append(PLAYER_X, TRACK_POINTS['0']['X'].T)
PLAYER_Y = TRACK_POINTS['1']['Y'].T
PLAYER_Y = np.append(PLAYER_Y, TRACK_POINTS['0']['Y'].T)

# Clip 2
# PLAYER_X = TRACK_POINTS['0']['X'].T
# PLAYER_X = np.append(PLAYER_X, TRACK_POINTS['1']['X'].T)
# PLAYER_Y = TRACK_POINTS['0']['Y'].T
# PLAYER_Y = np.append(PLAYER_Y, TRACK_POINTS['1']['Y'].T)


GT_df['tracking_x_s'] = PLAYER_X
GT_df['tracking_y_s'] = PLAYER_Y


GT_df['abs_error'] = np.sqrt(abs(GT_df['GT_X'] - GT_df['tracking_x'])**2 + abs(GT_df['GT_Y'] - GT_df['tracking_y'])**2)
GT_df['abs_error_meters'] = GT_df['abs_error'] * PIXELS2METERS

error_0_df = GT_df.loc[GT_df['ID'] == 0, ['frame', 'abs_error_meters']]
AVG_ERROR_0 = error_0_df['abs_error_meters'].mean()
error_1_df = GT_df.loc[GT_df['ID'] == 1, ['frame', 'abs_error_meters']]
AVG_ERROR_1 = error_1_df['abs_error_meters'].mean()

AVG_ERROR = GT_df['abs_error_meters'].mean()
print(GT_df.head())

plt.figure(figsize=(10, 5), dpi= 100)
plt.title("Absolute Error of Tracking Position and Ground Truth")
plt.plot(error_1_df['frame'], error_1_df['abs_error_meters'], 'r',
         linewidth=0.8, label="Absolute error - ID = 0")
plt.plot(error_0_df['frame'], error_0_df['abs_error_meters'], 'b',
         linewidth=0.8, label="Absolute error - ID = 1")
plt.plot([0, 1300], [AVG_ERROR, AVG_ERROR], 'k--',
         label="Average error = {}m".format(round(AVG_ERROR, 3)))
plt.legend(loc='upper left')
plt.xlabel('Frame'), plt.ylabel('Error (m)')
plt.xlim(0, 1300) #plt.ylim(0, 0.16)
plt.xticks(np.arange(0, 1300, step=120))
plt.axes().xaxis.set_minor_locator(ticker.MultipleLocator(60))
plt.savefig("results/plots/{}_error_plot.png".format(CLIP), bbox_inches='tight',
            facecolor='w', edgecolor='k')


GT_df['abs_error_s'] = np.sqrt(abs(GT_df['GT_X'] - GT_df['tracking_x_s'])**2 + abs(GT_df['GT_Y'] - GT_df['tracking_y_s'])**2)
GT_df['abs_error_s_meters'] = GT_df['abs_error_s'] * PIXELS2METERS

error_0_df = GT_df.loc[GT_df['ID'] == 0, ['frame', 'abs_error_s_meters']]
AVG_ERROR_0 = error_0_df['abs_error_s_meters'].mean()
error_1_df = GT_df.loc[GT_df['ID'] == 1, ['frame', 'abs_error_s_meters']]
AVG_ERROR_1 = error_1_df['abs_error_s_meters'].mean()

AVG_ERROR = GT_df['abs_error_s_meters'].mean()
print(GT_df.head())

plt.figure(figsize=(10, 5), dpi= 100)
plt.title("Absolute Error of Smoothed Tracking Position and Ground Truth")
plt.plot(error_1_df['frame'], error_1_df['abs_error_s_meters'], 'r',
         linewidth=0.8, label="Absolute error - ID = 0")
plt.plot(error_0_df['frame'], error_0_df['abs_error_s_meters'], 'b',
         linewidth=0.8, label="Absolute error - ID = 1")
plt.plot([0, 1300], [AVG_ERROR, AVG_ERROR], 'k--',
         label="Average error = {}m".format(round(AVG_ERROR, 3)))
plt.legend(loc='upper left')
plt.xlabel('Frame'), plt.ylabel('Error (m)')
plt.xlim(0, 1300) #plt.ylim(0, 0.16)
plt.xticks(np.arange(0, 1300, step=120))
plt.axes().xaxis.set_minor_locator(ticker.MultipleLocator(60))
plt.savefig("results/plots/{}_error_plot_smoothed.png".format(CLIP), bbox_inches='tight',
            facecolor='w', edgecolor='k')

# # Seperate plots 
# plt.figure(figsize=(10, 5), dpi= 100)
# plt.title("Absolute Error of Tracking Position and Ground Truth, ID = 1")
# plt.plot(error_1_df['frame'], error_1_df['abs_error_meters'], 'r',
         # linewidth=0.8, label="Absolute error")
# plt.plot([0, 846], [AVG_ERROR_1, AVG_ERROR_1], 'k--',
         # label="Average error = {}m".format(round(AVG_ERROR_1, 3)))
# plt.legend(loc='upper left')
# plt.xlabel('Frame'), plt.ylabel('Error (m)')
# plt.xlim(0, 846), plt.ylim(0, 0.16)
# plt.xticks(np.arange(0, 850, step=120))
# plt.axes().xaxis.set_minor_locator(ticker.MultipleLocator(60))
# plt.savefig("results/plots/{}_error_plot_1.png".format(CLIP), bbox_inches='tight',
            # facecolor='w', edgecolor='k')
