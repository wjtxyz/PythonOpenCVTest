import cv2
import utils
import numpy as np

img2 = None

curveList = []
averageLen = 10


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def getLaneCurve(img):
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    imgThres = utils.thresholding(img, np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max]))

    # imgResult = img.copy()

    hT, wT, c = img.shape

    points = utils.valTrackBars()
    img = utils.drawPoints(img.copy(), points)

    # h1, w1, c1 = imgWarp.shape
    # h2, w2, c2 = imgThres.shape
    # imgThres = cv2.cvtColor(imgThres, cv2.COLOR_GRAY2BGR);
    imgWarp = utils.warpImg(imgThres, points, wT, hT)

    middlePoint, imgHist = utils.histogram(imgWarp, display=True, minPer=0.5, region=4)
    averagePoint, imgHist = utils.histogram(imgWarp, display=True, minPer=0.9)
    print(averagePoint - middlePoint)
    curveRaw = averagePoint - middlePoint

    # de-noise curve value
    curveList.append(curveRaw)
    if len(curveList) > averageLen:
        curveList.pop(0)

    curve = int(sum(curveList) / len(curveList))

    imgInvWarp = utils.warpImg(imgWarp, points, wT, hT, inv=True)
    imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
    imgInvWarp[0:hT // 3, 0:wT] = 0, 0, 0

    imgLaneColor = np.zeros_like(img)
    imgLaneColor[:] = 0, 255, 0

    imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
    imgResult = cv2.addWeighted(img, 1, imgLaneColor, 1, 0)

    midY = 100
    cv2.putText(imgResult, str(curve), (wT // 2 - 80, 85), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255))

    # cv2.imshow('Result', imgResult)
    #
    #
    # cv2.imshow('Histogram', imgHist)

    # cv2.imshow('Thres', imgThres)
    cv2.imshow('Warp', imgWarp)
    cv2.imshow('Result', imgResult)
    # cv2.imshow('Vid', img)
    # imgStack =  np.hstack((img, imgWarp, imgThres))
    # cv2.imshow('Stack', imgStack)

    # print(f'{h_min}/{h_max}, {s_min}/{s_max}, {v_min}/{v_max}, {w}/{w1}, {h}/{h1}')
    return None


def empty(b):
    getLaneCurve(img2)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cap = cv2.VideoCapture('vid1.mp4')
    initialTrackBarVals = [280, 114, 350, 0]
    utils.initializeTrackBars(initialTrackBarVals, empty)

    while True:
        success, img2 = cap.read()
        if not success:
            pass
        getLaneCurve(img2)
        if -1 != cv2.waitKey(0):
            pass
