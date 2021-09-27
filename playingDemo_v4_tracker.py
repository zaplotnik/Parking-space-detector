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
import mosse

class Car():
    
    def __init__(self, ID, location, time):
        self.type = 'Car'
        self.ID = ID
        self.location = location                      # location on high res frame as [x1,y1,x2,y2]
        self.counted = False
        self.timesDetected = 1
        self.lastDetection = time
        self.isActive = True
#        self.tracker = None
        self.tracker = cv2.TrackerKCF_create()
        self.tracker_box = None                  # tracker box of low res frame as [x1_lr,y1_lr,w_lr,h_lr]
        self.tracker_location = None
        self.tracker_ok = False
        
    def updateLocation(self, location):
        self.location = location
        
    def updateCounted(self):
        self.counted = True
                
    def updateTimesDetected(self):
        self.timesDetected += 1
    
    def updatePassive(self):
        self.isActive = False
        
    def updateLastDetection(self, time):
        self.lastDetection = time
        
    def updateTrackerLocation(self,box,wr,hr):
        self.tracker_box = box
        self.tracker_location = (int(box[0]/wr),int(box[1]/hr),int((box[0]+box[2])/wr),int((box[1]+box[3])/hr))
    
    
    
        

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
              car.timesDetected, car.counted, round(car.lastDetection,1), car.isActive))


def updateCarStatus(t, t_threshold):
    """
    If car hasn't been found last t_trehshold, put it on passive.
    """
    for car in carsDetected:
        if t - car.lastDetection > t_threshold:
            car.updatePassive()

def get_subrectangle(x1,y1,x2,y2):
    """
    Get 1/9 center subrectangle of initial rectangle
    """
    diff_x = x2 - x1
    diff_y = y2 - y1
    x1_a = x1 +   diff_x/3
    x2_a = x1 + 2*diff_x/3
    y1_a = y1 +   diff_y/3
    y2_a = y1 + 2*diff_y/3
    
    return x1_a,y1_a,x2_a,y2_a

def orig2lr(x,y,wr,hr):
    return x*wr, y*hr
    
def lr2orig(x_lr,y_lr,wr,hr):
    return x_lr/wr,y_lr/hr

def orig2lr_int(x,y,wr,hr):
    return int(x*wr), int(y*hr)
    
def lr2orig_int(x_lr,y_lr,wr,hr):
    return int(x_lr/wr),int(y_lr/hr)

def rectangle_points_from_tracker(loc,wr,hr):
    """
    Transform low resolution tracker locations [x_lr,y_lr,dx_lr,dy_lr] to original scale rectangle points [int]
    """
    return lr2orig_int(loc[0],loc[1],wr,hr), lr2orig_int(loc[0]+loc[2],loc[1]+loc[3],wr,hr)

demoID = 3 # 1,2 or 3

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
color = (0,170,255)

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
confidenceLevel = 0.9  # 80% confidence level in DNN SSD
paintDetections = True # Drawing detections of DNN
showInfo = True # Showing program info
relative_threshold = 1000 # 200 pixel for distance 0.1, 24 pixel for distance 0.05

# Demo Parameter Initialization
iFrame = 0 # Counter
t = 0 # [sec] time counter
detection_interval  = 10 # DNN detection frame rate
tracking_interval = 2    # object tracking interval
carCounter = 0  # Counter of parked cars
carCounterL = 0  # Counter of parked cars - LEFT SIDE
carCounterR = 0  # Counter of parked cars - RIGHT SIDE
frameTime = 1./fpsRate # [sec] time between two consecutive frames
waitKeyTimer = 100 # For Demo Purposes: Fast Forward, no waiting, OVERWRITE

carsDetected = [] # object library initialization
remove_list = []  # list of cars to be removed from carsDetected
# object: [objectID, last location, already counted, side, times detected]
# ie. objects.append[1, coords, False, side, 1]
detCounter = 0
carID = 0

h,w = 700,1200
h_lr,w_lr = 300,426

hr = 1.*h_lr/h
wr = 1.*w_lr/w

start_time = time.time()
while (cap.isOpened()):
    ret, frame = cap.read()
#    (h, w) = frame.shape[:2]
    frame_low_res = cv2.resize(frame, (w_lr, h_lr))
#    (h_lr,w_lr) = frame_low_res.shape[:2]
    
    # remove inactive cars from list - must be done in place and not passed to function (for speed)
    remove_list = []    
    for (iCar,car) in enumerate(carsDetected):
        if not car.isActive:
            remove_list.append(iCar)
    for ind in remove_list[::-1]:
        del carsDetected[ind]
    
    # update car tracking  (now, all cars are active)
    if iFrame % tracking_interval == 0:
        
        for car in carsDetected:
            car.tracker_ok,tracker_box = car.tracker.update(frame_low_res)
            car.updateTrackerLocation(tracker_box,wr,hr)
                        
            if not car.tracker_ok:
                car.isActive = False
            else:
                if paintDetections:
                    cv2.rectangle(frame, car.tracker_location[:2], car.tracker_location[2:], [0,60,255], 2)
                    cv2.circle(frame, ((car.tracker_location[0]+car.tracker_location[2])//2,(car.tracker_location[1]+car.tracker_location[3])//2),10,[0,60,255],-1)
            
    
    if iFrame % detection_interval == 0:
        
        ## !! RESIZE TAKES A LOT OF TIME - IMPORTANT TO PIPE LOW-RES VIDEO, e.g. 640x480
        blob = cv2.dnn.blobFromImage(frame_low_res, 0.007843, (426, 300), 127.5)
#        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (600, 600)), 0.007843, (600, 600), 127.5)
        net.setInput(blob)
        detections = net.forward()   
        detCounter += 1         
        nDet = np.arange(0, detections.shape[2])    
        
        # loop over the detections
        for i in nDet:
            # check if detection is car
            idx = int(detections[0, 0, i, 1])
            if idx == 7:
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]
                
                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence > confidenceLevel:
                    # Object Detected!

                    # compute the (x, y)-coordinates of the bounding box for the object 
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")
                    cX, cY, side, distance = getOrientation(box, w, h)
                    
                    if paintDetections:
                        # display the detected object
                        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                        print("{0}: [INFO SSD] {1}, side {2}, center ({3}, {4}), distance {5}.".format(\
                              round(t,1), label, side, cX, cY, distance))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                        cv2.circle(frame, (cX,cY),20,color,-1)
                        y = y1 - 15 if y1 - 15 > 15 else y1+ 15
                        cv2.putText(frame, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Check if the car is already known: Check last distance and last location
                    det = False
                    nCars = len(carsDetected)
                    if nCars > 0:
                        for (iCar,car) in enumerate(carsDetected):#loop through all cars
                            if car.isActive:
                                car_centerX, car_centerY, car_side, car_distance = getOrientation(car.tracker_location, w, h)
                                (x1_tracker,y1_tracker,x2_tracker,y2_tracker) = car.tracker_location
                                cX_tracker = (x2_tracker+x1_tracker)/2
                                cY_tracker = (y2_tracker+y1_tracker)/2
                            else:
                                continue
                            
                            # check if detected car is already tracked - smart checks
                            checksum = 0
                            # check 1 - size relative distance of centers
                            print cX_tracker,cX,"             ", cY_tracker,cY
                            if abs(cX_tracker - cX)/distance < relative_threshold and abs(cY_tracker - cY)/distance < relative_threshold:
#                            and x1 < cX_tracker < x2 and y1 < cY_tracker < y2 :
                                print 1                                
                                checksum += 1
                            
                                car.updateLocation(box)
                                car.updateTimesDetected()
                                car.updateLastDetection(t)
                                
                                # reinitialize tracker
                                x1_lr,y1_lr = orig2lr_int(x1,y1,wr,hr)
                                x2_lr,y2_lr = orig2lr_int(x1,y1,wr,hr)
                                tracker_box = (x1_lr,y1_lr,x2_lr-x1_lr,y2_lr-y1_lr)
                                car.tracker.clear()
                                car.tracker.init(frame_low_res,tracker_box)
                                car.updateTrackerLocation(tracker_box,wr,hr)
                                det = True
                                if car.timesDetected > 0 and not car.counted: # This car is candidate to be parked
                                    car.updateCounted()
                                    carCounter += 1
#                                    if side == 0: # left
#                                        carCounterL += 1
#                                    else:
#                                        carCounterR += 1
                                break
                        if not det:
                            # Save object in database if its not in the database yet
                            carID += 1    
                            newCar = Car(carID, box, t)
                            x1_lr,y1_lr = orig2lr_int(x1,y1,wr,hr)
                            x2_lr,y2_lr = orig2lr_int(x1,y1,wr,hr)
                            tracker_box = (x1_lr,y1_lr,x2_lr-x1_lr,y2_lr-y1_lr)
                            newCar.tracker.init(frame_low_res,tracker_box)
                            newCar.updateTrackerLocation(tracker_box,wr,hr)
                            newCar.updateCounted()
                            carsDetected.append(newCar)
                            carCounter += 1
                            

                    else: # First car
                        # Save object in database if its not in the database yet
                        carID += 1
                        newCar = Car(carID, box, t)
                        x1_lr,y1_lr = orig2lr_int(x1,y1,wr,hr)
                        x2_lr,y2_lr = orig2lr_int(x2,y2,wr,hr)
                        tracker_box = (x1_lr,y1_lr,x2_lr-x1_lr,y2_lr-y1_lr)
                        newCar.tracker.init(frame_low_res,tracker_box)
                        newCar.updateTrackerLocation(tracker_box,wr,hr)
                        newCar.updateCounted()
#                        print box,tracker_box, newCar.tracker_location
                        carsDetected.append(newCar)
                        
                        carCounter += 1
     
             
                
    # how long to wait will depend on our driving speed. 
    # If fast, forget cars soon, if we're stuck in traffic jam or waiting at the stop sign, increase temporary to infinity        
    updateCarStatus(t, 2.)
        
    if showInfo: # Show DEMO info: ID, Location, Max Spaces, Counted Spaces
        #putText(Mat& img, const string& text, Point org, int fontFace, double fontScale, Scalar color, int thickness=1, int lineType=8, bool bottomLeftOrigin=false )
        cv2.rectangle(frame, (0,0), (230,195), (204,229,255), thickness = -1)
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
        cv2.putText(frame, 'CarCounter: ' + str(carCounter), (5, 175), infoFont, infoFontSize, infoColor, infoFontThick)
        
    # Finalize Frame
    iFrame += 1 
    t += frameTime
    #print(iFrame, len(nDet))
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

carStatus()