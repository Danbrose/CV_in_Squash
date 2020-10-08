#!/usr/bin/env python3
"""
Author: Daniel Ambrose
Created: 2020-09-15 16:13:17
Description: This is a test file
"""
import json
import glob
import sys
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from scipy import stats, integrate
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

# print(frame_data[0]['0'][5][1])

# df = pd.DataFrame(
    # {"player_x" : [frame_data[0]['0'][5][0], frame_data[1]['1'][5][0]],
     # "player_y" : [frame_data[0]['0'][5][1], frame_data[1]['1'][5][1]],
     # "box_origin" : [frame_data[0]['0'][0], frame_data[1]['1'][0]],
     # "width" : [frame_data[0]['0'][1], frame_data[1]['1'][1]],
     # "height" : [frame_data[0]['0'][2], frame_data[1]['1'][2]],
     # "centroid" : [frame_data[0]['0'][3], frame_data[1]['1'][3]],
     # "box_colour" : [frame_data[0]['0'][4], frame_data[1]['1'][4]]},
    # index = pd.MultiIndex.from_tuples(
        # [('0', 1), ('0', 2)], names = ['frame', 'player_ID']))

def translate_point(x,y,h):
    denom = h[2, 0] *x + h[2, 1] * y + h[2, 2]
    xPrime = (h[0, 0] *x + h[0, 1] * y + h[0, 2])/ denom
    yPrime = (h[1, 0] *x + h[1, 1] * y + h[1, 2]) / denom
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
TRACK_POINTS['0']['X'] = savgol_filter(TRACK_POINTS['0']['X'], INTERVAL,
                                       POLY).round().astype(int)
TRACK_POINTS['0']['Y'] = savgol_filter(TRACK_POINTS['0']['Y'], INTERVAL,
                                       POLY).round().astype(int)
TRACK_POINTS['1']['X'] = savgol_filter(TRACK_POINTS['1']['X'], INTERVAL,
                                       POLY).round().astype(int)
TRACK_POINTS['1']['Y'] = savgol_filter(TRACK_POINTS['1']['Y'], INTERVAL,
                                       POLY).round().astype(int)

PLAYER_0_X = TRACK_POINTS['0']['X']
PLAYER_0_Y = TRACK_POINTS['0']['Y']
PLAYER_1_X = TRACK_POINTS['1']['X']
PLAYER_1_Y = TRACK_POINTS['1']['Y']


PIXELS2METERS = 6.4/775
df_0 = pd.DataFrame(
    {"track_point_x" : PLAYER_0_X.T,
     "track_point_y" : PLAYER_0_Y.T})
df_1 = pd.DataFrame(
    {"track_point_x" : PLAYER_1_X.T,
     "track_point_y" : PLAYER_1_Y.T})

df_0['travel'] = np.sqrt(df_0['track_point_x'].diff().fillna(0)**2 +
df_0['track_point_y'].diff().fillna(0)**2)
df_1['travel'] = np.sqrt(df_1['track_point_x'].diff().fillna(0)**2 +
df_1['track_point_y'].diff().fillna(0)**2)
# court coverage
CC_0 = round(df_0['travel'].sum() * PIXELS2METERS)
CC_1 = round(df_1['travel'].sum() * PIXELS2METERS)
print(CC_0, CC_1)

court_plot = plt.imread("court_plot.png")

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(court_plot)
ax = sns.kdeplot(PLAYER_0_X, PLAYER_0_Y,
                 cmap="coolwarm",
                 alpha=0.85,
                 shade=True,
                 shade_lowest=False,
                 levels=50,
                 antialiased=True,
                 cbar=True,
                 cbar_kws={'ticks':[], 'shrink': 0.9})

ax.set_title('Player 0', fontsize='x-large')
plt.axis('off')
plt.xlim(0, 520)
plt.ylim(775, 0)
plt.annotate('Court Coverage: {}m'.format(CC_0), (155, 40), fontsize='small')

ax = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(court_plot)
ax = sns.kdeplot(PLAYER_1_X, PLAYER_1_Y,
                 cmap="coolwarm",
                 bw_adjust=2,
                 alpha=0.85,
                 shade=True,
                 shade_lowest=False,
                 levels=50,
                 antialiased=True,
                 cbar=True,
                 cbar_kws={'ticks':[], 'shrink': 0.9})

ax.set_title('Player 1', fontsize='x-large')
# imgplot.set_clim(0.0, 0.7)
# plt.colorbar(cmap, "coolwarm", ticks=[0, 1], orientation='vertical')
plt.axis('off')
plt.xlim(0, 520)
plt.ylim(775, 0)
plt.annotate('Court Coverage: {}m'.format(CC_1), (155, 40), fontsize='small')
plt.subplots_adjust(wspace=-0.25, hspace=0)

fig = ax.get_figure()
fig.savefig('results/plots/match_1_rally_2_heatmap.png', transparent=True,
            bbox_inches='tight', pad_inches=0.1)
