import cv2
import numpy as np

def read_bgr(file_bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def specular_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)

    # 반사광 특징: V 아주 밝음, S 낮음(채도↓)
    v_th = (v > 230).astype(np.uint8)
    s_th = (s < 40).astype(np.uint8)

    # 그 외 하이라이트도 잡도록 그레이 임계
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, g_th = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_or(v_th*255, s_th*255)
    mask = cv2.bitwise_or(mask, g_th)

    # 모폴로지로 다듬기
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def inpaint_glare(bgr, mask):
    # Telea 또는 NS
    radius = 3
    repaired = cv2.inpaint(bgr, (mask>0).astype(np.uint8)*255, radius, cv2.INPAINT_TELEA)
    # 대비 보정(문서에 유리)
    lab = cv2.cvtColor(repaired, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    repaired = cv2.cvtColor(cv2.merge([l2,a,b]), cv2.COLOR_LAB2BGR)
    return repaired

def detect_document_quad(bgr):
    """
    문서 외곽 사각형 추정 (있으면 시점보정)
    """
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
    return best  # (4,2) or None

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
    # 타겟 크기 추정
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
    """
    여러 프레임에서 글레어 마스크를 이용해 글레어 영역만 다른 프레임 픽셀로 대체.
    간단한 노출 정합으로 깜빡임 줄임.
    """
    base = frames_bgr[0].copy()
    base_lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB)
    l_base, a_base, b_base = cv2.split(base_lab)

    for idx in range(1, len(frames_bgr)):
        f = frames_bgr[idx]
        mask = specular_mask(base)
        if mask.mean() < 1: 
            continue
        # 밝기 정합
        f_lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
        l_f, a_f, b_f = cv2.split(f_lab)
        gain = (l_base[l_base>0].mean() + 1e-6)/(l_f[l_base>0].mean() + 1e-6)
        f_corr = cv2.cvtColor(cv2.merge([
            np.clip(l_f*gain,0,255).astype(np.uint8), a_f, b_f
        ]), cv2.COLOR_LAB2BGR)
        # 마스크 부위만 교체( feathering )
        mask_f = cv2.GaussianBlur(mask, (21,21), 0).astype(np.float32)/255.0
        base = (mask_f[:,:,None]*f_corr + (1-mask_f)[:,:,None]*base).astype(np.uint8)

    return base

def pipeline(files_bytes):
    """
    files_bytes: List[bytes] (1개 또는 다중 프레임)
    returns: processed_bgr, merged_used(bool)
    """
    frames = [read_bgr(b) for b in files_bytes]
    if len(frames) > 1:
        img = merge_multi_frames(frames)
        merged = True
    else:
        img = frames[0]
        merged = False

    # 글레어 제거 (단일/병합 결과 공통)
    mask = specular_mask(img)
    repaired = inpaint_glare(img, mask)

    # 문서 시점 보정 (있을 때만)
    quad = detect_document_quad(repaired)
    if quad is not None:
        repaired = warp_document(repaired, quad)

    return repaired, merged
