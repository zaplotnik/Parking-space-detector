# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:47:03 2015

@author: Jan
"""



import sys
#sys.path.append(r'F:\KardioBit\dev\ParkingSpace')
sys.path.append("/home/ziga/kardiobit/ParkingSpace/")
import numpy as np
#import dlib
import cv2
import time

def getOrientation(coords, w, h):
    """ Gets LEFT or RIGHT side of the car center location.
    Inputs: 
        array cords: array of coordinates [startX, startY, endX, endY]
        int w: width of the frame
    Output: 
        int centerX, X coordinate of object orientation
        int centerX, Y coordinate of object orientation
        int side, 0 = LEFT, 1 = RIGHT
        int distance, relative size of w and h of the object"""
    startX, startY, endX, endY = coords
    centerX = int((startX + endX)/2)
    centerY = int((startY + endY)/2)
    side = 1
    if centerX < w/2.:
        side = 0
    size = round((startX - endX) * (startY - endY) / (w * h), 3)
    return centerX, centerY, side, size
    

demoID = 1 # 1,2 or 3

# Demo database
demoDB = {1: {'MaxParkingSpotsL': 0, 'MaxParkingSpotsR': 18, 'StreetName': 'Prekmurska ulica'},
          2: {'MaxParkingSpotsL': 16, 'MaxParkingSpotsR': 23, 'StreetName': 'Stihova ulica'},
          3: {'MaxParkingSpotsL': 14, 'MaxParkingSpotsR': 14, 'StreetName': 'Vurnikova ulica'}}

clipName = 'TestShort'+str(demoID)+'.mp4'
#cap = cv2.VideoCapture(r'F:\KardioBit\dev\ParkingSpace\TestShort3.mp4')
cap = cv2.VideoCapture(clipName)

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


# load our serialized model from disk
print("[INFO SSD] Loading model...")
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
#tracker = dlib.correlation_tracker()

fpsRate = cap.get(5)
try:
    waitKeyTimer = int(1./fpsRate * 1000)
except:
    waitKeyTimer = 50
print(waitKeyTimer, fpsRate)


# Settings
infoColor = (0, 0, 0)
infoFontSize = 0.5
infoFontThick = 1
infoFont = cv2.FONT_HERSHEY_SIMPLEX
confidenceLevel = 0.8  # 80% confidence level in DNN SSD

paintDetections = True # Drawing detections of DNN
showInfo = True # Showing program info

# Demo Parameter Initialization
iFrame = 0 # Counter
t = 0 # [sec] time counter
detection_interval  = 10 # DNN detection frame rate
carCounter = 0  # Counter of parked cars
carCounterL = 0  # Counter of parked cars - LEFT SIDE
carCounterR = 0  # Counter of paqrked cars - RIGHT SIDE
frameTime = 1./fpsRate # [sec] time between two consecutive frames
waitKeyTimer = 100 # For Demo Purposes: Fast Forward, no waiting, OVERWRITE

objects = [] # object library initialization
# object: [objectID, last location, already counted, side, times detected]
# ie. objects.append[1, coords, False, side, 1]
trackers = []

car_id = 0
start_time = time.time()
while (cap.isOpened()):
    ret, frame = cap.read()
    (h, w) = frame.shape[:2]
                       
    if iFrame  % detection_interval == 0:
        
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()            
        
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]
            
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > confidenceLevel:
                # Object Detected!
                
                # extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for
                # the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                if startX<0: startX = 0 
                if startY<0: startY = 0 
                if endX > w: endX = w 
                if endY > h: endY = h 
                centerX, centerY, side, size = getOrientation(box, w, h)

                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                print("{0}: [INFO SSD] {1}, side {2}, center ({3}, {4}), relative_size {5}.".format(round(t,1), label, side, centerX, centerY, size))
                     
                
                # no object exists
                if objects == [] and size > 0.08: 
                    car_id += 1
                    location = ((startX, startY, endX, endY),(centerX,centerY),size)
                    objects.append([car_id,location,1,label])
                
                # some object exists
                else:
                    for (j,object_) in enumerate(objects):
                        [car_id, location_old, times_detected,label] = object_
                        ((startX_old, startY_old, endX_old, endY_old),(centerX_old,centerY_old),size_old) = location_old
                        
                        # compare new location with previous object location
                        # if object is small (i.e. far away), locations needs to be closer to previous             
                        relative_center_distance = np.sqrt((centerX_old-centerX)**2 + (centerY_old-centerY)**2) / size
                        
                        print relative_center_distance                        
                        
#                        # compute % of overlap
#                        SA = 1.*(endX_old-startX_old)*(endY_old-startY_old)
#                        SB = 1.*(endX-startX)*(endY-startY)
#                        SI = 1.*max(0,min(endX_old,endX) - max(startX_old,startX)) *  max(0,min(endY_old,endY) - max(startY_old,startY))
#                        SU = SA + SB - SI
#                        relative_overlap = 1.*SI/SU
#                        
#                        # location from detector
#                        location = ((startX, startY, endX, endY),(centerX,centerY))
#
#                        # car already tracked
##                        if relative_cdeistance < 80 and SI/SU > 0.5 and 0.7 < size / size_old < 1.3:
##                            # update only location and times_detected
##                            objects[j][1] = location
##                            objects[j][2] += 1
#                        
#                        # new car
#                        else:
#                            if distance > 0.05:
#                                car_id += 1
#                                objects.append([car_id,location,1,label])
#                                trackers.append(cv2.TrackerKCF_create())
#                                ok = trackers[-1].init(frame,(startX, startY, endX-startX, endY-startY))
                           
    if paintDetections: 
        for (j,object_) in enumerate(objects):  
            print iFrame, j, object_
            ((startX, startY, endX, endY),(centerX,centerY),size) = object_[1]
            label = object_[-1]
            
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            cv2.circle(frame, (centerX,centerY),5,COLORS[idx],-1)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)   

         
    if showInfo: # Show DEMO info: ID, Location, Max Spaces, Counted Spaces
        #putText(Mat& img, const string& text, Point org, int fontFace, double fontScale, Scalar color, int thickness=1, int lineType=8, bool bottomLeftOrigin=false )
        cv2.rectangle(frame, (0,0), (230,175), (204,229,255), thickness = -1)
        cv2.putText(frame, 'Demo: ' + str(demoID), (5, 15), infoFont, infoFontSize, infoColor, infoFontThick)
        cv2.putText(frame, 'Street: ' + demoDB[demoID]['StreetName'], (5, 35), infoFont, infoFontSize, infoColor, infoFontThick)
        cv2.putText(frame, 'MaxParkingSpotsLeft: ' + str(demoDB[demoID]['MaxParkingSpotsL']), (5, 55), infoFont, infoFontSize, infoColor, infoFontThick)
        cv2.putText(frame, 'ParkedCarsCounterLeft: ' + str(carCounterL), (5, 75), infoFont, infoFontSize, infoColor, infoFontThick)
        cv2.putText(frame, 'MaxParkingSpotsRight: ' + str(demoDB[demoID]['MaxParkingSpotsR']), (5, 95), infoFont, infoFontSize, infoColor, infoFontThick)
        cv2.putText(frame, 'ParkedCarsCounterRight: ' + str(carCounterR), (5, 115), infoFont, infoFontSize, infoColor, infoFontThick)
        cv2.putText(frame, 'Video Time: ' + str(round(t,1)), (5, 135), infoFont, infoFontSize, infoColor, infoFontThick)
        real_time = time.time()-start_time
        if real_time < t: infoColor_time = (0,255,0)
        else: infoColor_time = (0,0,255)
        cv2.putText(frame, 'Real Time: ' + str(round(real_time,1)), (5, 155), infoFont, infoFontSize, infoColor_time, infoFontThick)
 
    # Finalize Frame
    iFrame += 1 
    t += frameTime
    try:
        cv2.imshow('Demo', frame)
        if cv2.waitKey(waitKeyTimer) & 0xFF == ord('q'):
            break
        if cv2.waitKey(waitKeyTimer) & 0xFF == ord('p'):
            while 1:
               if cv2.waitKey(waitKeyTimer) & 0xFF == ord('p'):
                   break
    except:
        break
    

cap.release()
cv2.destroyAllWindows()