import cv2

def apply_clahe(image):
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(yuv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y = clahe.apply(y)
    yuv = cv2.merge((y, cr, cb))
    return cv2.cvtColor(yuv, cv2.COLOR_YCrCb2BGR)

# 추가적인 유틸리티 함수들...