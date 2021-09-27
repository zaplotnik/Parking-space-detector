# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:24:49 2017

@author: ziga
"""

import cv2
import sys
import mosse
import dlib
 
if __name__ == '__main__' :
 
    # Set up tracker.
    # Instead of MIL, you can also use
 
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[2]
 
    
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
#    tracker = dlib.correlation_tracker()
    # Read video
    video = cv2.VideoCapture("TestShort3.mp4")
 
    # Exit if video not opened.
    if not video.isOpened():
        print "Could not open video"
        sys.exit()
 
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print 'Cannot read video file'
        sys.exit()
     
    # Define an initial bounding box
#    bbox = (287, 23, 86, 320)
 
    # Uncomment the line below to select a different bounding box
#    bbox = cv2.selectROI(frame, False)
 
    # Initialize tracker with first frame and bounding box
    
    i = 0
    selected=False
    while True:
        # Read a new frame
        ok, frame = video.read()
        frame = cv2.resize(frame, (426, 300))
#        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not ok:
            break
         
        # Start timer
        timer = cv2.getTickCount()
        
        # CHOOSE WHERE TO START
        if i > 250:
            if selected == False:
                bbox = cv2.selectROI(frame, False)
#                tracker.start_track(frame, dlib.rectangle(bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]))
                ok = tracker.init(frame, bbox)
#                tracker = mosse.MOSSE(frame_gray, [bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]])
                selected = True
            # Update tracker
            if i % 2 == 0:
                ok, bbox = tracker.update(frame)
                print ok,bbox
#                tracker.update(frame_gray)
            
#                ok = True
                tracker.update(frame)
            
     
            # Draw bounding box
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
#                a = tracker.get_position()
#                print (a.right()-a.left())*(a.bottom()-a.top())
#                cv2.rectangle(frame, (int(a.left()),int(a.top())), (int(a.right()),int(a.bottom())), (255,0,0), 2, 1)
#                cv2.rectangle(frame,tracker.box()[0],tracker.box()[1],(255,0,0), 2, 1)
            else:
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
     
            # Display tracker type on frame
            cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
        
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
 
        # Display result
        cv2.imshow("Tracking", frame)
        i += 1
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break