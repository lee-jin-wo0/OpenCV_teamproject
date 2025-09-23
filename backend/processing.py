import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# 기존 processing.py의 모든 함수 (read_bgr, specular_mask, inpaint_glare 등)
def read_bgr(file_bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def specular_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)

    v_th = (v > 230).astype(np.uint8)
    s_th = (s < 40).astype(np.uint8)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, g_th = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_or(v_th*255, s_th*255)
    mask = cv2.bitwise_or(mask, g_th)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def inpaint_glare(bgr, mask):
    radius = 3
    repaired = cv2.inpaint(bgr, (mask>0).astype(np.uint8)*255, radius, cv2.INPAINT_TELEA)
    lab = cv2.cvtColor(repaired, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    repaired = cv2.cvtColor(cv2.merge([l2,a,b]), cv2.COLOR_LAB2BGR)
    return repaired

def detect_document_quad(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    area_img = w*h
    best = None; best_area = 0
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > 0.2*area_img and area > best_area:
                best_area = area
                best = approx.reshape(-1,2).astype(np.float32)
    return best

def order_corners(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)[:,0]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl,tr,br,bl], dtype=np.float32)

def warp_document(bgr, quad):
    quad = order_corners(quad)
    w1 = np.linalg.norm(quad[1]-quad[0])
    w2 = np.linalg.norm(quad[2]-quad[3])
    h1 = np.linalg.norm(quad[3]-quad[0])
    h2 = np.linalg.norm(quad[2]-quad[1])
    W = int(max(w1,w2))
    H = int(max(h1,h2))
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(bgr, M, (W,H))

def merge_multi_frames(frames_bgr):
    base = frames_bgr[0].copy()
    base_lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB)
    l_base, a_base, b_base = cv2.split(base_lab)
    for idx in range(1, len(frames_bgr)):
        f = frames_bgr[idx]
        mask = specular_mask(base)
        if mask.mean() < 1: 
            continue
        f_lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
        l_f, a_f, b_f = cv2.split(f_lab)
        gain = (l_base[l_base>0].mean() + 1e-6)/(l_f[l_base>0].mean() + 1e-6)
        f_corr = cv2.cvtColor(cv2.merge([
            np.clip(l_f*gain,0,255).astype(np.uint8), a_f, b_f
        ]), cv2.COLOR_LAB2BGR)
        mask_f = cv2.GaussianBlur(mask, (21,21), 0).astype(np.float32)/255.0
        base = (mask_f[:,:,None]*f_corr + (1-mask_f)[:,:,None]*base).astype(np.uint8)
    return base

# --- homomorphic_filter_module.py의 함수 추가 ---
def homomorphic_filter_yuv(img, sigma=10, gamma1=0.3, gamma2=1.5):
    # ... (기존 homomorphic_filter_yuv 함수 내용)
    img_YUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y = img_YUV[:,:,0]
    rows, cols = y.shape
    imgLog = np.log1p(np.array(y, dtype='float') / 255)
    M, N = 2*rows + 1, 2*cols + 1
    (X, Y) = np.meshgrid(np.linspace(0, N-1, N), np.linspace(0, M-1, M))
    Xc, Yc = np.ceil(N/2), np.ceil(M/2)
    gaussianNumerator = (X - Xc)**2 + (Y - Yc)**2
    LPF = np.exp(-gaussianNumerator / (2*sigma*sigma))
    HPF = 1 - LPF
    LPF_shift, HPF_shift = np.fft.ifftshift(LPF.copy()), np.fft.ifftshift(HPF.copy())
    img_FFT = np.fft.fft2(imgLog.copy(), (M, N))
    img_LF = np.real(np.fft.ifft2(img_FFT.copy() * LPF_shift, (M, N)))
    img_HF = np.real(np.fft.ifft2(img_FFT.copy() * HPF_shift, (M, N)))
    img_adjusting = gamma1*img_LF[0:rows, 0:cols] + gamma2*img_HF[0:rows, 0:cols]
    img_exp = np.expm1(img_adjusting)
    img_exp = (img_exp - np.min(img_exp)) / (np.max(img_exp) - np.min(img_exp))
    img_out = np.array(255*img_exp, dtype = 'uint8')
    img_YUV[:,:,0] = img_out
    result = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)
    return result

# --- image_utils.py의 함수들 ---
# (이전 답변에서 복사한 함수들을 그대로 사용)
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

# --- 수정된 pipeline 함수 ---
def pipeline(files_bytes):
    frames = [read_bgr(b) for b in files_bytes]
    if len(frames) > 1:
        img = merge_multi_frames(frames)
        merged = True
    else:
        img = frames[0]
        merged = False

    # homomorphic 필터 최적화
    gamma1_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    gamma2_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    best_image = None
    best_text_length = -1
    
    # OCR을 바로 실행하는 것이 아니라, 텍스트가 잘 인식되는지 평가해야 함.
    # 이를 위해 OCR 로직을 processing.py로 임시로 가져와서 테스트.
    # 실제로는 main.py에서 OCR을 호출해야 하므로 이 부분은 개념적으로만 이해.
    from ocr import run_ocr
    
    for g1 in gamma1_values:
        for g2 in gamma2_values:
            filtered_image = homomorphic_filter_yuv(img.copy(), gamma1=g1, gamma2=g2)
            # OCR 결과를 평가하여 최적의 이미지 선택
            try:
                text, _ = run_ocr(filtered_image)
                if len(text) > best_text_length:
                    best_text_length = len(text)
                    best_image = filtered_image
            except Exception as e:
                # 오류 발생 시 해당 감마값 조합은 건너뛰기
                continue

    # 최적의 이미지로 OCR 재실행
    if best_image is not None:
        img = best_image
    else:
        # 최적의 이미지를 찾지 못하면 기본값 사용
        img = homomorphic_filter_yuv(img, gamma1=0.3, gamma2=1.5)

    # 글레어 제거 (단일/병합 결과 공통)
    mask = specular_mask(img)
    repaired = inpaint_glare(img, mask)

    # 문서 시점 보정 (있을 때만)
    quad = detect_document_quad(repaired)
    if quad is not None:
        repaired = warp_document(repaired, quad)

    return repaired, merged