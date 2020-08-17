# credit for all the code below https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/
#%%
import cv2
import sys
import matplotlib.pyplot as plt
import time
#(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')￼

if __name__ == '__main__' :
    
    tracker_types = [
            'BOOSTING',
            'MIL',
            'KCF',
            'TLD',
            'MEDIANFLOW',
            #'GOTURN',
            'MOSSE',
            'CSRT'
            ]
    
    for i, tracker_type in enumerate(tracker_types):
        start = time.time()

        pathOut = "results/Tracking/{0}.avi".format(tracker_type)
        
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()

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
        
        # Define an initial bounding box
        bbox = (1370, 528, 117, 246)

        # Initialize tracker with first frame and bounding box
        ok = tracker.init(frame, bbox)

        frame_array = []
        while True:
            # Read a new frame
            ok, frame = video.read()
            if not ok:
                break
            
            # Start timer
            timer = cv2.getTickCount()

            # Update tracker
            ok, bbox = tracker.update(frame)

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

            # Draw bounding box
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            else :
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100,110), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

            # Display tracker type on frame
            cv2.putText(frame, tracker_type + " Tracker", (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)    
            
            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

            #Display result
            # plt.imshow(frame, interpolation = 'bicubic')
            # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            # plt.show()
            height, width, layers = frame.shape
            size = (width,height)
            
            #inserting the frames into an image array
            frame_array.append(frame)
            
        out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), 60, size)
        
        for i in range(len(frame_array)):
            # writing to a image array
            out.write(frame_array[i])

        # Release everything if job is finished
        out.release()
        end = time.time()
        duration = end - start
        print([tracker_type, duration])
# %%
