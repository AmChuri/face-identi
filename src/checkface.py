import numpy as np
import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascades/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascades/haarcascade_smile.xml')

while(True):
    #capture camera frame
    ret, frame = cap.read()

    cv2.imshow('frame',frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
