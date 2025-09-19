import cv2
import numpy as np

def imclearborder(imgBW, radius):
    imgBWcopy = imgBW.copy()
    contours,hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    imgRows, imgCols = imgBW.shape[0], imgBW.shape[1]
    contourList = []
    for idx in np.arange(len(contours)):
        cnt = contours[idx]
        for pt in cnt:
            if pt is None or len(pt) == 0 or pt[0] is None:
                continue
            rowcnt, colcnt = pt[0][1], pt[0][0]
            check1 = (rowcnt >= 0 and rowcnt < radius) or (rowcnt >= imgRows-1-radius and rowcnt < imgRows)
            check2 = (colcnt >= 0 and colcnt < radius) or (colcnt >= imgCols-1-radius and colcnt < imgCols)
            if check1 or check2:
                contourList.append(idx)
                break
    for idx in contourList:
        cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)
    return imgBWcopy

def bwareaopen(imgBW, areaPixels):
    imgBWcopy = imgBW.copy()
    contours,hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for idx in np.arange(len(contours)):
        area = cv2.contourArea(contours[idx])
        if (area >= 0 and area <= areaPixels):
            cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)
    return imgBWcopy
