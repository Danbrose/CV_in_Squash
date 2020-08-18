#%%
import json
import itertools
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.patches as patches


# insert the keypoints and the image
n = "001"
this = "results/mamatch_1_rally_1_1080_60fps_BODY_25_MaxPeople.json/match_1_rally_1_1080_60fps_000000000{0}_keypoints.json".format(n)
that = "data/frames/match_1_rally_1_1080_60fps/image-{0}.jpeg".format(n)

with open(this) as json_file:
    data = json.load(json_file)

#extracts the pose keypoints from the json
person1_KP = data['people'][0]['pose_keypoints_2d']
person2_KP = data['people'][1]['pose_keypoints_2d']

# Removes any zeros from the data which will skew the data
# Zeros are returned for keypoints that have not been detected
while (person1_KP.count(0)): 
    person1_KP.remove(0)  
while (person2_KP.count(0)): 
    person2_KP.remove(0)

# init an empty list to reformat the data into a useable format
person1_KP_new = []
person2_KP_new = []
for i in range(0, int(len(person1_KP)/3)):
    x = [person1_KP.pop(-3), person1_KP.pop(-2), person1_KP.pop(-1)]
    person1_KP_new.append(x)
for i in range(0, int(len(person2_KP)/3)):
    y = [person2_KP.pop(-3), person2_KP.pop(-2), person2_KP.pop(-1)]
    person2_KP_new.append(y)

# init empty list to split x and y into seperate lists
x_KP1 = []
y_KP1 =[]
x_KP2 = []
y_KP2 =[]

# enumerates through the key points and splits x and y for both person1 and 2
for n, value in enumerate(person1_KP_new):
    x_KP1.append(person1_KP_new[n][0])
    y_KP1.append(person1_KP_new[n][1])
for n, value in enumerate(person2_KP_new):
    x_KP2.append(person2_KP_new[n][0])
    y_KP2.append(person2_KP_new[n][1])

# putting x and y lists into another list... too many lists going on here
KP1 = [x_KP1, y_KP1]
KP2 = [x_KP2, y_KP2]

# Function to find the centroid of the keypoints
# Calcualtes the appropriate width and height of the bounding box depending on the
# distsances to the furthest points in x and y from the centroid
def bbox_from_KP(KP, sf):
    sum_x = np.sum(KP[0])
    sum_y = np.sum(KP[1])
    x_c = round(sum_x/len(KP[0]))
    y_c = round(sum_y/len(KP[1]))
    centroid = (x_c, y_c)
    width = round (max( abs( x_c-max(KP[0]) ), abs( x_c-min(KP[0]) ) ) * sf)
    height = round (max( abs( y_c-max(KP[1]) ), abs( y_c-min(KP[1]) ) ) * sf)
    # Due to the centroid often being located slightly lower than the hips,
    # the box is not exactly centered about the centroid and instead shifted up 
    # slightly to avoid cutting off the head 
    box_origin = ( round(x_c-width/1.9), round(y_c-height/1.9) )
    return width, height, box_origin, centroid

# calls the fuction to calculate the bounding boxes
width1, height1, box_origin1, centroid1 = bbox_from_KP(KP1, 2.2)
width2, height2, box_origin2, centroid2 = bbox_from_KP(KP2, 2.2)

# create a figure to display the image and the keypoints overlayed
figure(num=None, figsize=(11, 10), dpi=200, facecolor='w', edgecolor='k')
im = plt.imread(that)
implot = plt.imshow(im, origin='upper')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
for n, value in enumerate(person1_KP_new):
    plt.plot(person1_KP_new[n][0], person1_KP_new[n][1], 'bo')
for n, value in enumerate(person2_KP_new):
    plt.plot(person2_KP_new[n][0], person2_KP_new[n][1], 'ro')

# draws ractagles using the calculated values
rect1 = plt.Rectangle(
    (box_origin1[0], box_origin1[1]),
    width1, height1,
    fill=False, edgecolor='blue')
rect2 = plt.Rectangle(
    (box_origin2[0], box_origin2[1]),
    width2, height2,
    fill=False, edgecolor='red')

print(rect1)
print(rect2)

# Plots rectagles on the figure
plt.gca().add_patch(rect1)
plt.gca().add_patch(rect2)

# Plots the centroids for clarity
plt.plot(centroid1[0], centroid1[1], 'wo')
plt.plot(centroid2[0], centroid2[1], 'wo')
#%%










# %%
