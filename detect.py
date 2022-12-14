# -*- coding: utf-8 -*-

 # -*- coding: utf-8 -*-


# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 48
 
# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0
def rect_to_bb(rect):
    x=rect.left;()
    y=rect.top();
    w=rect.right()-x;
    h=rect.bottom()-y;
def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
 
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
 
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
 
    # return the eye aspect ratio
    return ear
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# grab the indexes of the facial landmarks for the left and
# right eye, respectively

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = cv2.VideoCapture('test.mov')
fileStream = True
print('d')
# vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
# fileStream = False
#time.sleep(1.0)


# loop over frames from the video stream
while True:
    # if this is a file video stream, then we need to check if
    # there any more frames left in the buffer to process
    print('a')
    
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    print('b')

    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    ret,frame = vs.read()
    print('c')
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        
        shape = face_utils.shape_to_np(shape)
        
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        
        x=leftEye[0][0]
        y=leftEye[1][1]
        w=leftEye[3][0]-leftEye[0][0]
        h=leftEye[4][1]-leftEye[1][1]
       
        ax=(x+w)/2
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
 
        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        


        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
       
     
        
        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1
 
        # otherwise, the eye aspect ratio is not below the blink
        # threshold
        else:
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
                #cv2.rectangle(frame, (x,y),(x+w,y+h), (0, 255, 0), 1)
                #cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                
                eye=frame[y:y+h,x:x+w]
                eye=cv2.resize(eye,(300,150),interpolation=cv2.INTER_AREA)
                eye_gray=cv2.cvtColor(eye,cv2.COLOR_BGR2GRAY)
                eye_blur=cv2.GaussianBlur(eye_gray,(7,7),0)
                
                #_,thre=cv2.threshold(eye_blur,127,255,cv2.THRESH_BINARY_INV)
                th, thre = cv2.threshold(eye_gray, 100, 255,cv2.THRESH_BINARY_INV) 
                cv2.imshow("Threshold", thre)
                #count,_=cv2.findContours(eye_blur,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                contours, _ = cv2.findContours(thre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
                for cnt in contours:
                    (x,y,w,h)=cv2.boundingRect(cnt)
                    cv2.rectangle(eye,(x,y),(x+w,y+h),(0,0,255),3)
                    px=(x+w)/2
                    py=(y+h)/2
                    if px<ax:
                        print("left")
                       
                        break;
                    else:
                        print("right")
                        
                        break;
                cv2.imshow("EyeBlur", eye_blur) #Burred
                cv2.imshow("Eye", eye)
                    # reset the eye frame counter
            COUNTER = 0
            
            

        # draw the total number of blinks on the frame along with
        # the computed eye aspect ratio for the frame
        cv2.putText(frame, "Blinks: {}".format(COUNTER), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        break;
    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
