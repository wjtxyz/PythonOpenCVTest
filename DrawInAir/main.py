import cv2
import numpy as np


def empty(b):
    pass


cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 480)
cv2.createTrackbar("Hue Min", "TrackBars", 100, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 200, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)

cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

imgResult = None
myPoints = []


def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = 0, 0, 0, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            # cv2.drawContours(imgResult, cnt, -1, (255, 0, 0), 3)
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            x, y, w, h = cv2.boundingRect(approx)
    return x + w // 2, y


def filterImage(img):
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    imgInRange = cv2.inRange(imgHsv, lower, upper)
    x, y = getContours(imgInRange)
    if x != 0 and y != 0:
        myPoints.append([x, y])

    for point in myPoints:
        cv2.circle(imgResult, (point[0], point[1]), 10, (255, 255, 0), cv2.FILLED)

    cv2.circle(imgResult, (x, y), 5, (255, 0, 0), cv2.FILLED)

    return cv2.cvtColor(imgInRange, cv2.COLOR_GRAY2BGR)


frameWidth = 320
frameHeight = 240
cap = cv2.VideoCapture(1)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 130)
while True:
    success, img = cap.read()
    imgResult = img.copy()
    imgStack = np.hstack((filterImage(img), imgResult))
    cv2.imshow("ImageStack", imgStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break
