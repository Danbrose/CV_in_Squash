#%%
import json
import itertools
import numpy as np
import cv2
import sys
import time

start = time.time()

pathOut = "bbox_openpose.avi"

# Read video
video = cv2.VideoCapture("match_1_rally_1_1080_60fps.mp4")

# Exit if video not opened.
if not video.isOpened():
    print( "Could not open video" )
    sys.exit()

# Read first frame.
ok, frame = video.read()
if not ok:
    print( 'Cannot read video file' )
    sys.exit()

frame_array = []
n = "0000"
while True:
        
    key_points = "results/mamatch_1_rally_1_1080_60fps_BODY_25_MaxPeople.json/match_1_rally_1_1080_60fps_00000000{0}_keypoints.json".format(n)
    with open(key_points) as json_file:
        data = json.load(json_file)

    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break

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

    p1 = (box_origin1[0], box_origin1[1])
    p2 = (box_origin1[0]+width1 , box_origin1[1]+height1)
    cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        
    height, width, layers = frame.shape
    size = (width,height)
    
    #inserting the frames into an image array
    frame_array.append(frame)
    n = str(int(n) + 1)
    print(n)

out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), 60, size)

for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])

    # Release everything if job is finished
out.release()
end = time.time()
duration = end - start
 
    