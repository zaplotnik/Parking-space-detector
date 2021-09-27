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
        self.center = ((self.location[0]+self.location[2])/2,(self.location[1]+self.location[3])/2)
        self.counted = False
        self.timesDetected = 1
        self.lastDetection = time
        self.isActive = True
        self.center_tracker = None
        self.center_tracker_hres = None
        
    def updateLocation(self, location):
        self.location = location
        self.center = ((location[0]+location[2])/2,(location[1]+location[3])/2)
        
    def updateCounted(self):
        self.counted = True
                
    def updateTimesDetected(self):
        self.timesDetected += 1
    
    def updatePassive(self):
        self.isActive = False
        
    def updateLastDetection(self, time):
        self.lastDetection = time
        
    def updateTrackerLocation(self,c,wr,hr):
        self.center_tracker = c
        self.center_tracker_hres = (c[0]/wr,c[1]/hr)
    
    
    
        

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

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

demoID = 1 # 1,2 or 3

# Demo database
demoDB = {1: {'MaxParkingSpotsL': 0, 'MaxParkingSpotsR': 18, 'StreetName': 'Prekmurska ulica'},
          2: {'MaxParkingSpotsL': 16, 'MaxParkingSpotsR': 23, 'StreetName': 'Stihova ulica'},
          3: {'MaxParkingSpotsL': 14, 'MaxParkingSpotsR': 14, 'StreetName': 'Vurnikova ulica'}}

clipName = 'TestShort'+str(demoID)+'.mp4'
#cap = cv2.VideoCapture(r'F:\KardioBit\dev\ParkingSpace\TestShort3.mp4')


# Improving video processing data
# https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
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
paintDetections = True# Drawing detections of DNN
showInfo = True # Showing program info
relative_threshold = 1000 # 200 pixel for distance 0.1, 24 pixel for distance 0.05

# Demo Parameter Initialization
iFrame = 0 # Counter
t = 0 # [sec] time counter
detection_interval  = 6 # DNN detection frame rate
tracker_interval=2
carCounter = 0  # Counter of parked cars
carCounterL = 0  # Counter of parked cars - LEFT SIDE
carCounterR = 0  # Counter of parked cars - RIGHT SIDE
frameTime = 1./fpsRate # [sec] time between two consecutive frames
waitKeyTimer = 1 # For Demo Purposes: Fast Forward, no waiting, OVERWRITE

carsDetected = [] # object library initialization
remove_list = []  # list of cars to be removed from carsDetected
# object: [objectID, last location, already counted, side, times detected]
# ie. objects.append[1, coords, False, side, 1]
detCounter = 0
carID = 0

h,w = 720,1280
h_lr,w_lr = 72,128 # 72,128, 90,160

hr = 1.*h_lr/h
wr = 1.*w_lr/w


ret, frame = cap.read()
prevgray = cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(w_lr,h_lr))

start_time = time.time()
while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),(w_lr,h_lr))
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.2, 3, 5, 3, 7, 1.5, 0)
    prevgray = gray
    
    # remove inactive cars from list - must be done in place and not passed to function (for speed)
    remove_list = []    
    for (iCar,car) in enumerate(carsDetected):
        if not car.isActive:
            remove_list.append(iCar)
    for ind in remove_list[::-1]:
        del carsDetected[ind]

    # update car centers according to optical flow vector   
    if iFrame % tracker_interval == 0:
        for car in carsDetected:
            # call previous center
            car_cx,car_cy = car.center_tracker
    
            # find closest optical flow vector
            u,v = flow[int(car_cy),int(car_cx),:]
            new_car_cx = car_cx + u
            new_car_cy = car_cy + v
            
            # when tracker center gets near the frame boundary, change car to passive
            if new_car_cx < 5 or new_car_cx > w_lr-5 or new_car_cy < 5 or new_car_cy > h_lr-5:  
                car.updatePassive()
            else:
                cv2.circle(frame,(int(new_car_cx/wr),int(new_car_cy/hr)),15,(0,60,255),-1)
                car.updateTrackerLocation((new_car_cx,new_car_cy),wr,hr)  
             
        
    # update center tracking  (now, all cars are active)
    if iFrame % detection_interval == 0:
        
        ## !! RESIZE TAKES A LOT OF TIME - IMPORTANT TO PIPE LOW-RES VIDEO, e.g. 640x480
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (227, 227)), 0.007843, (227, 227), 127.5)
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
                    cx, cy, side, car_size = getOrientation(box, w, h)
                    
                    if paintDetections:
                        # display the detected object
                        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                        print("{0}: [INFO SSD] {1}, side {2}, center ({3}, {4}), distance {5}.".format(\
                              round(t,1), label, side, cx, cy, car_size))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                        cv2.circle(frame, (cx,cy),20,color,-1)
                        y = y1 - 15 if y1 - 15 > 15 else y1+ 15
                        cv2.putText(frame, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Check if the car is already known
                    det = False
                    nCars = len(carsDetected)
                    if nCars > 0:
                        for (iCar,car) in enumerate(carsDetected):#loop through all cars
                            if car.isActive:
                                cx_tracker,cy_tracker = car.center_tracker_hres
                            else:
                                continue
                            
                            # check if detected car is already tracked - smart checks
#                            checksum = 0
                            # check 1 - size relative distance of centers
                            print cx_tracker,cx,"             ", cy_tracker,cy
                            if abs(cx_tracker - cx)/car_size < relative_threshold and abs(cy_tracker - cy)/car_size < relative_threshold:
#                            and x1 < cX_tracker < x2 and y1 < cY_tracker < y2 :
                                print 1                                
#                                checksum += 1
                            
                                car.updateLocation(box)
                                car.updateTimesDetected()
                                car.updateLastDetection(t)
                                
                                # now start tracker from new detected center transformed to low resolution
                                car.updateTrackerLocation((cx*wr,cy*hr),wr,hr) #UPDATE TRACKER !!!!
                                
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
                            
                            #UPDATE TRACKER !!!!                            
                            newCar.updateTrackerLocation((cx*wr,cy*hr),wr,hr)
                            newCar.updateCounted()
                            carsDetected.append(newCar)
                            carCounter += 1
                            

                    else: # First car
                        # Save object in database if its not in the database yet
                        carID += 1
                        newCar = Car(carID, box, t)

                        #UPDATE TRACKER !!!!
                        newCar.updateTrackerLocation((cx*wr,cy*hr),wr,hr)
                        newCar.updateCounted()
#                        print box,tracker_box, newCar.tracker_location
                        carsDetected.append(newCar)
                        
                        carCounter += 1
     
             
                
    # how long to wait will depend on our driving speed. 
    # If fast, forget cars soon, if we're stuck in traffic jam or waiting at the stop sign, increase temporary to infinity        
    updateCarStatus(t, 1.)
        
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
    
print time.time()-start_time
cap.release()
cv2.destroyAllWindows()

carStatus()