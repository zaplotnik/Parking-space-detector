# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:47:03 2015

@author: Jan
"""
import sys
sys.path.append(r'F:\KardioBit\dev\ParkingSpace')
import numpy as np
import cv2
import time

class Car():
    
    def __init__(self, ID, location, time):
        self.type = 'Car'
        self.ID = ID
        self.location = location
        self.counted = False
        self.timesDetected = 1
        self.lastDetection = time
        self.stilActive = True
        
    def updateLocation(self, location):
        self.location = location
        
    def updateCounted(self):
        self.counted = True
                
    def updateTimesDetected(self):
        self.timesDetected += 1
    
    def updatePassive(self):
        self.stilActive = False
        
    def updateLastDetection(self, time):
        self.lastDetection = time
        
        

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
    distance = round((startX - endX) * (startY - endY) / (w * h), 3)
    return centerX, centerY, side, distance


def carStatus():
    """ 
    Show me status of detected cars.
    """    
    for car in carsDetected:
        print('Car ID: {0}, detected {1} times. Counted: {2}, last detection {3}, active {4}.'.format(car.ID, \
              car.timesDetected, car.counted, round(car.lastDetection,1), car.stilActive))


def updateCarStatus(t, t_threshold):
    """
    If car hasn't been found last t_trehshold, put it on passive.
    """
    for car in carsDetected:
        if t - car.lastDetection > t_threshold:
            car.updatePassive()

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
confidenceLevel = 0.95  # 80% confidence level in DNN SSD
paintDetections = True # Drawing detections of DNN
showInfo = False # Showing program info
relative_threshold = 2000 # 120 pixel for distance 0.1, 24 pixel for distance 0.05

# Demo Parameter Initialization
iFrame = 0 # Counter
t = 0 # [sec] time counter
detection_interval  = 5 # DNN detection frame rate
carCounter = 0  # Counter of parked cars
carCounterL = 0  # Counter of parked cars - LEFT SIDE
carCounterR = 0  # Counter of parked cars - RIGHT SIDE
frameTime = 1./fpsRate # [sec] time between two consecutive frames
waitKeyTimer = 1 # For Demo Purposes: Fast Forward, no waiting, OVERWRITE

carsDetected = [] # object library initialization
# object: [objectID, last location, already counted, side, times detected]
# ie. objects.append[1, coords, False, side, 1]
detCounter = 0
nDet = 0

start_time = time.time()
while (cap.isOpened()):
    ret, frame = cap.read()

    if ret and iFrame % detection_interval == 0:
        # initialize a tracker on the first frame
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (227, 227)), 0.007843, (227, 227), 127.5)
        net.setInput(blob)
        detections = net.forward()   
        detCounter += 1         
        nDet = np.arange(0, detections.shape[2])    

        if paintDetections:
             # loop over the detections
            for i in nDet:
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]
                
                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence > confidenceLevel:
                    # Object Detected!
                    carCounter += 1
                    
                    
                    # extract the index of the class label from the `detections`,
                    # then compute the (x, y)-coordinates of the bounding box for
                    # the object
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    centerX, centerY, side, distance = getOrientation(box, w, h)
                    # display the prediction
                    label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                    print("{0}: [INFO SSD] {1}, side {2}, center ({3}, {4}), distance {5}.".format(round(t,1), label, side, centerX, centerY, distance))
                    cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                    
                    # Check if the car is already known: Check last distance and last location
                    det = False
                    if len(carsDetected) > 0 and distance > 0.04:
                        for car in carsDetected:#go through all cars
                            if car.stilActive:
                                car_centerX, car_centerY, car_side, car_distance = getOrientation(car.location, w, h)
                            else:
                                continue
                            if abs(car_centerX - centerX)/distance < relative_threshold and abs(car_centerY - centerY)/distance < relative_threshold and car_side == side and 0.7 < car_distance / distance < 1.3:
                                # this car is already in database, update its position and count times
                                car.updateLocation(box)
                                car.updateTimesDetected()
                                car.updateLastDetection(t)
                                det = True
                                if car.timesDetected > 1 and not car.counted: # This car is candidate to be parked
                                    car.updateCounted()
                                    if side == 0: # left
                                        carCounterL += 1
                                    else:
                                        carCounterR += 1
                                break
                        if not det:
                            # Save object in database if its not in the database yet
                            carsDetected.append(Car(len(carsDetected), box, t))
                            print len(carsDetected)
                    elif distance > 0.04: # First car
                        # Save object in database if its not in the database yet
                        carsDetected.append(Car(len(carsDetected), box, t))
                
    updateCarStatus(t, 0.3)
        
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
    if iFrame == 670: print time.time()-start_time    
    # Finalize Frame
    iFrame += 1 
    t += frameTime
    #print(iFrame, len(nDet))
    try:
#        cv2.imshow('Demo', frame)
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

carStatus()