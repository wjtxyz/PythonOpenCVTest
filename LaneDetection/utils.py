import numpy as np
import cv2


def thresholding(img, lowerWhite=np.array([0, 0, 0]), upperWhite=np.array([255, 255, 255])):
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower = np.array([30, 0, 110])
    # upper = np.array([255, 255, 255])
    maskWhite = cv2.inRange(imgHsv, lowerWhite, upperWhite)
    return maskWhite


def warpImg(img, points, w, h, inv = False):
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2, pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, dsize=(w, h))
    # print(f'{pts1}******{pts2}*****{matrix}')
    return imgWarp


def empty(b):
    pass


def initializeTrackBars(initialTrackVals, refreshCB, wT=480, hT=270):
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 360, 480)
    cv2.createTrackbar("Hue Min", "TrackBars", 8, 179, refreshCB)
    cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, refreshCB)
    cv2.createTrackbar("Val Min", "TrackBars", 167, 255, refreshCB)

    cv2.createTrackbar("Hue Max", "TrackBars", 138, 179, refreshCB)
    cv2.createTrackbar("Sat Max", "TrackBars", 40, 255, refreshCB)
    cv2.createTrackbar("Val Max", "TrackBars", 210, 255, refreshCB)

    cv2.createTrackbar("Width Top", "TrackBars", initialTrackVals[0], wT, refreshCB)
    cv2.createTrackbar("Height Top", "TrackBars", initialTrackVals[1], hT, refreshCB)
    cv2.createTrackbar("Width Bottom", "TrackBars", initialTrackVals[2], wT, refreshCB)
    cv2.createTrackbar("Height Bottom", "TrackBars", initialTrackVals[3], hT, refreshCB)


def valTrackBars(wT=480, hT=270):
    widthTop = cv2.getTrackbarPos("Width Top", "TrackBars")
    heightTop = cv2.getTrackbarPos("Height Top", "TrackBars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "TrackBars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "TrackBars")

    points = np.float32([((wT - widthTop) // 2, heightTop), ((wT + widthTop) // 2, heightTop),
                         ((wT - widthBottom) // 2, hT - heightBottom), ((wT + widthBottom) // 2, hT - heightBottom)])
    return points


def drawPoints(img, points):
    for i in range(4):
        cv2.circle(img, (int(points[i][0]), int(points[i][1])), 10, (0, 0, 255), cv2.FILLED)
    return img


def histogram(img, minPer=0.1, display=False, region=1):
    if region == 1:
        histVal = np.sum(img, axis=0)
    else:
        histVal = np.sum(img[img.shape[0] // region:, :], axis=0)
    # print(histVal)
    maxValue = np.max(histVal)
    minVaue = minPer * maxValue

    indexArray = np.where(histVal > minVaue)
    basePoint = int(np.average(indexArray))
    print(basePoint)

    if display:
        imgHist = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        for x, intensity in enumerate(histVal):
            cv2.line(imgHist, (x, img.shape[0]), (x, img.shape[0] - intensity // 255 // region), (255, 0, 255), 1)
        cv2.circle(imgHist, (basePoint, img.shape[0]), 15, (255, 255, 0), cv2.FILLED)
        return basePoint, imgHist
    return basePoint
