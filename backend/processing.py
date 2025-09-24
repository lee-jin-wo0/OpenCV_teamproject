import cv2
import numpy as np

# =========================
# 전역 튜닝(필요 시 조정)
# =========================
GRID = {
    "blockSize": 25,    # 3x3 격자 검출용 적응 이진화 블록 (홀수)
    "C": 15,            # 임계 보정
    "pad_ratio": 0.03,  # 셀 크롭 패딩
    "min_grid_area": 0.10,
}


# =========================
# 보조 유틸
# =========================
def _cap_size(bgr, max_side=2000):
    """긴 변을 max_side로 캡(속도/정확 균형)"""
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    scale = max_side / float(m)
    return cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _sharpen_and_boost(bgr):
    """언샤프 + LAB CLAHE로 글자 대비/선명도 강화"""
    blur = cv2.GaussianBlur(bgr, (0, 0), 1.0)
    sharp = cv2.addWeighted(bgr, 1.4, blur, -0.4, 0)
    lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
    return out


# =========================
# I/O
# =========================
def read_bgr(file_bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = _cap_size(img, max_side=2000)
    return img


# =========================
# 일반 글레어 마스크/인페인트
# =========================
def specular_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v_th = (v > 230).astype(np.uint8) * 255
    s_th = (s < 40).astype(np.uint8) * 255

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, g_th = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_or(v_th, s_th)
    mask = cv2.bitwise_or(mask, g_th)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return mask


def inpaint_glare(bgr, mask):
    radius = 2
    repaired = cv2.inpaint(bgr, (mask > 0).astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)

    lab = cv2.cvtColor(repaired, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    repaired = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
    return repaired


# =========================
# 화이트보드 특화(파란 마커 보호)
# =========================
def _blue_ink_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (90, 60, 40), (130, 255, 255))
    m2 = cv2.inRange(hsv, (100, 40, 30), (140, 255, 255))
    m = cv2.bitwise_or(m1, m2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    return m  # 0/255


def _illumination_flatten(bgr, kernel=65):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if kernel % 2 == 0:
        kernel += 1
    bg = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    bg = np.maximum(bg, 1)
    norm = (gray.astype(np.float32) * (gray.mean() / bg)).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)


def specular_mask_whiteboard(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v_th = (v > 230).astype(np.uint8) * 255
    s_th = (s < 40).astype(np.uint8) * 255
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, g_th = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_or(v_th, s_th)
    mask = cv2.bitwise_or(mask, g_th)

    # 글자 보호
    ink = _blue_ink_mask(bgr)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(ink))

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def inpaint_glare_whiteboard(bgr, mask):
    if mask.mean() < 1:
        return bgr
    feather = cv2.GaussianBlur(mask, (21, 21), 0).astype(np.float32) / 255.0
    repaired = cv2.inpaint(bgr, (mask > 0).astype(np.uint8) * 255, 2, cv2.INPAINT_TELEA)
    out = (feather[:, :, None] * repaired + (1 - feather)[:, :, None] * bgr).astype(np.uint8)
    return out


def _is_whiteboard_scene(bgr):
    ink = _blue_ink_mask(bgr)
    ink_ratio = (ink > 0).mean()
    bg_mean = bgr.mean()
    return (bg_mean > 150) and (ink_ratio > 0.001)


# =========================
# 문서 검출/시점 보정
# =========================
def detect_document_quad(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    area_img = w * h
    best = None
    best_area = 0
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > 0.2 * area_img and area > best_area:
                best_area = area
                best = approx.reshape(-1, 2).astype(np.float32)
    return best  # (4,2) or None


def warp_document(bgr, quad, min_side=64, max_side=2200):
    """견고한 투시 보정(비정상 수치/크기 폴백)"""
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return bgr
    if quad is None or len(quad) != 4:
        return bgr

    def _order(pts):
        pts = np.array(pts, dtype=np.float32).reshape(-1, 2)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1)[:, 0]
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]
        bl = pts[np.argmax(d)]
        return np.stack([tl, tr, br, bl], axis=0)

    quad = _order(quad)

    w1 = np.linalg.norm(quad[1] - quad[0])
    w2 = np.linalg.norm(quad[2] - quad[3])
    h1 = np.linalg.norm(quad[3] - quad[0])
    h2 = np.linalg.norm(quad[2] - quad[1])
    if not np.isfinite([w1, w2, h1, h2]).all():
        return bgr

    W = float(max(w1, w2))
    H = float(max(h1, h2))
    if W < 1 or H < 1:
        return bgr

    # 최소/최대 사이즈 보정
    if W < min_side or H < min_side:
        s = max(min_side / W, min_side / H)
        W *= s; H *= s
    if W > max_side or H > max_side:
        s = min(max_side / W, max_side / H)
        W *= s; H *= s

    W = int(round(W)); H = int(round(H))
    if W < min_side or H < min_side or W > 10000 or H > 10000:
        return bgr

    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    if not np.isfinite(M).all() or abs(np.linalg.det(M[:, :2])) < 1e-6:
        return bgr

    warped = cv2.warpPerspective(
        bgr, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    warped = np.clip(warped, 0, 255).astype(np.uint8)
    warped = np.ascontiguousarray(warped)
    return warped


# =========================
# 다중 프레임 병합(글레어 영역 교체)
# =========================
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
        denom = (l_f[l_base > 0].mean() + 1e-6)
        gain = (l_base[l_base > 0].mean() + 1e-6) / denom
        l_corr = np.clip(l_f * gain, 0, 255).astype(np.uint8)
        f_corr = cv2.cvtColor(cv2.merge([l_corr, a_f, b_f]), cv2.COLOR_LAB2BGR)

        mask_f = cv2.GaussianBlur(mask, (21, 21), 0).astype(np.float32) / 255.0
        base = (mask_f[:, :, None] * f_corr + (1 - mask_f)[:, :, None] * base).astype(np.uint8)

    return base


# =========================
# 파이프라인 (화이트보드 특화)
# =========================
def pipeline(files_bytes):
    frames = [read_bgr(b) for b in files_bytes]
    img = merge_multi_frames(frames) if len(frames) > 1 else frames[0]

    # 1) 조명 평탄화(반사/그림자 완화)
    flat = _illumination_flatten(img, kernel=65)

    # 2) 화이트보드 감지 → 전용 마스크/인페인트
    if _is_whiteboard_scene(flat):
        mask = specular_mask_whiteboard(flat)
        repaired = inpaint_glare_whiteboard(flat, mask)
    else:
        mask = specular_mask(flat)
        repaired = inpaint_glare(flat, mask)

    # 3) 문서 시점 보정(가능하면)
    quad = detect_document_quad(repaired)
    if quad is not None:
        repaired = warp_document(repaired, quad)

    # 4) 마지막 품질 부스팅 + 인코딩 안전화
    repaired = _sharpen_and_boost(repaired)
    repaired = np.clip(repaired, 0, 255).astype(np.uint8)
    repaired = np.ascontiguousarray(repaired)

    # 만약 크기가 비정상이면 원본으로 폴백
    if repaired.size == 0 or repaired.shape[0] < 10 or repaired.shape[1] < 10:
        repaired = img

    merged_used = len(frames) > 1
    return repaired, merged_used


# =========================
# 3x3 격자 셀 추출 (책표지 등일 때만 사용)
# =========================
def find_grid_cells_3x3(bgr, debug=False):
    img = bgr.copy()
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bin_ = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, GRID["blockSize"], GRID["C"]
    )

    def morph_lines(src, ksize):
        k = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
        er = cv2.erode(src, k, iterations=1)
        di = cv2.dilate(er, k, iterations=2)
        return di

    horiz = morph_lines(bin_, (max(10, w // 30), 1))
    vert  = morph_lines(bin_, (1, max(10, h // 30)))
    grid = cv2.bitwise_and(horiz, vert)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grid = cv2.dilate(grid, k, iterations=1)

    cnts, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []

    area_img = w * h
    best = None; best_area = 0
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            a = cv2.contourArea(approx)
            if a > GRID["min_grid_area"] * area_img and a > best_area:
                best_area = a; best = approx.reshape(-1, 2).astype(np.float32)
    if best is None:
        return []

    def order(pts):
        s = pts.sum(axis=1); d = np.diff(pts, axis=1)[:, 0]
        tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]; bl = pts[np.argmax(d)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    quad = order(best)
    W = int(max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])))
    H = int(max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])))
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    warp = cv2.warpPerspective(img, M, (W, H))

    pad_x = int(GRID["pad_ratio"] * W); pad_y = int(GRID["pad_ratio"] * H)
    cell_w = W // 3; cell_h = H // 3
    cells = []
    for r in range(3):
        for c in range(3):
            x1 = max(c * cell_w + pad_x // 2, 0)
            y1 = max(r * cell_h + pad_y // 2, 0)
            x2 = min((c + 1) * cell_w - pad_x // 2, W)
            y2 = min((r + 1) * cell_h - pad_y // 2, H)
            if x2 - x1 > 10 and y2 - y1 > 10:
                cells.append(warp[y1:y2, x1:x2].copy())

    return cells if len(cells) >= 6 else []