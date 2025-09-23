# backend/ocr.py
import os, re
import numpy as np
import cv2
from paddleocr import PaddleOCR

# ---------------------------
# 전역 튜닝 포인트 (숫자만 바꿔가며 시험)
# ---------------------------
TUNE = {
    "det_db_box_thresh": 0.40,     # [0.30~0.60] 낮추면 작은 글자↑(잡음↑)
    "det_db_unclip_ratio": 2.0,    # [1.6~2.2] 박스 여유
    "det_limit_side_len": 1920,    # [1280~1920] 길수록 정확↑/느림↑
    "max_text_length": 96,         # [64~128] 긴 라벨이면 ↑
    "upscale_factor": 1.7,         # [1.4~1.8] 멀티패스 업스케일
    "use_binary_second_pass": True,
    "rotate_degrees": [0, 180, 90, 270],  # 방향 불안정 시 유용
    "tile_refine_upscale": 2.0,    # 박스별 재인식 업스케일
}

# ---------------------------
# 서버 모델(대형) 사용 옵션
#  - 환경변수나 경로가 있으면 자동 사용
# ---------------------------
USE_SERVER_MODELS = True
DET_DIR = os.getenv("PPOCR_DET_DIR", "").strip()      # 예: ./models/ch_ppocr_server_v2.0_det_infer
REC_DIR = os.getenv("PPOCR_REC_DIR", "").strip()      # 예: ./models/korean_PP-OCRv4_server_rec_infer
CLS_DIR = os.getenv("PPOCR_CLS_DIR", "").strip()      # (선택)

OCR_KW = dict(
    use_angle_cls=True,
    lang="korean",
    det_db_box_thresh=TUNE["det_db_box_thresh"],
    det_db_unclip_ratio=TUNE["det_db_unclip_ratio"],
    det_limit_side_len=TUNE["det_limit_side_len"],
    max_text_length=TUNE["max_text_length"],
    show_log=False,
)

if USE_SERVER_MODELS and DET_DIR and REC_DIR:
    OCR_KW.update({
        "det_model_dir": DET_DIR,
        "rec_model_dir": REC_DIR,
    })
    if CLS_DIR:
        OCR_KW["cls_model_dir"] = CLS_DIR

_ocr = PaddleOCR(**OCR_KW)


# ---------------------------
# 보조: 파싱/이진화/업스케일
# ---------------------------
def _parse_result(result):
    lines, boxes = [], []
    if not result:
        return "", [], 0.0
    for page in result or []:
        if not page: 
            continue
        for it in page:
            try:
                if isinstance(it, (list, tuple)) and len(it) == 2:
                    pts, data = it
                    if not pts or len(pts) != 4: 
                        continue
                    txt, conf = "", 0.0
                    if isinstance(data, (list, tuple)) and len(data) >= 2:
                        txt, conf = data[0], float(data[1])
                    elif isinstance(data, dict):
                        txt = data.get("text", "")
                        conf = float(data.get("score", 0.0))
                    P = []
                    for p in pts:
                        if isinstance(p, (list, tuple)) and len(p) == 2:
                            P.append([float(p[0]), float(p[1])])
                    if len(P) != 4: 
                        continue
                    boxes.append({"text": txt or "", "conf": conf, "box": P})
                    if txt: 
                        lines.append(txt)
            except Exception:
                continue
    mean_conf = float(np.mean([b["conf"] for b in boxes])) if boxes else 0.0
    return "\n".join(lines).strip(), boxes, mean_conf

def _run_once(image_bgr):
    res = _ocr.ocr(image_bgr, cls=True)
    return _parse_result(res)

def _binarize(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3,3), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

def _best_of(candidates):
    best = ("", [], 0.0)
    for im in candidates:
        t, b, m = _run_once(im)
        if (m > best[2]) or (m == best[2] and len(t) > len(best[0])):
            best = (t, b, m)
    return best


# ---------------------------
# 핵심1: 전체 이미지 멀티패스 (회전/업스케일/이진화)
# ---------------------------
def run_ocr(image_bgr):
    cand = []
    for deg in TUNE["rotate_degrees"]:
        im = image_bgr if deg == 0 else np.ascontiguousarray(np.rot90(image_bgr, k=deg//90))
        cand.append(im)
        if TUNE["use_binary_second_pass"]:
            cand.append(_binarize(im))
        uf = TUNE["upscale_factor"]
        if uf and uf > 1.0:
            h, w = im.shape[:2]
            up = cv2.resize(im, (int(w*uf), int(h*uf)), interpolation=cv2.INTER_CUBIC)
            cand.append(up)

    text, boxes, mean_conf = _best_of(cand)

    # 핵심2: 박스별 타일 재인식 (크롭→업스케일→이진화)
    if boxes:
        improved = []
        H, W = image_bgr.shape[:2]
        scale_up = TUNE["tile_refine_upscale"]
        for b in boxes:
            pts = np.array(b["box"], dtype=np.float32)
            x1 = max(int(np.min(pts[:,0]))-2, 0)
            y1 = max(int(np.min(pts[:,1]))-2, 0)
            x2 = min(int(np.max(pts[:,0]))+2, W-1)
            y2 = min(int(np.max(pts[:,1]))+2, H-1)
            crop = image_bgr[y1:y2+1, x1:x2+1]
            if crop.size < 10: 
                improved.append(b); 
                continue

            tiles = [crop]
            if scale_up and scale_up > 1.0:
                ch, cw = crop.shape[:2]
                tiles.append(cv2.resize(crop, (int(cw*scale_up), int(ch*scale_up)), interpolation=cv2.INTER_CUBIC))
            if TUNE["use_binary_second_pass"]:
                tiles += [_binarize(t) for t in list(tiles)]

            # 인식은 crop 단위에서만: det은 불필요하므로 그대로 _run_once 사용(과감히)
            best_txt, best_conf = b["text"], b["conf"]
            for timg in tiles:
                # 인식만 하고 싶지만 API 제약상 전체로 돌림 → crop이 작아 부담 적음
                tt, bb, mm = _run_once(timg)
                # crop에서 나오는 첫 문자열만 취함
                if tt and (mm >= best_conf or (abs(mm-best_conf) < 1e-3 and len(tt) > len(best_txt))):
                    best_txt, best_conf = tt, mm
            improved.append({**b, "text": best_txt, "conf": float(best_conf)})
        boxes = improved
        # 전체 텍스트도 박스 순으로 재구성
        text = " ".join([re.sub(r"\s+", " ", b["text"]).strip() for b in boxes if b["text"]]).strip()

    # 핵심3: 혼동 문자 보정 + 규칙 후처리
    text, boxes = _postprocess(text, boxes)
    return text, boxes


# ---------------------------
# 후처리(혼동 보정/규칙)
# ---------------------------
CONFUSION_MAP = str.maketrans({
    "０":"0","１":"1","２":"2","３":"3","４":"4","５":"5","６":"6","７":"7","８":"8","９":"9",
    "O":"0","o":"0","S":"5","s":"5","I":"1","l":"1","B":"8","Z":"2",
})

def _fix_confusions(s: str) -> str:
    s2 = s.translate(CONFUSION_MAP)
    # 연속 특수문자 정리
    s2 = re.sub(r"[|]{2,}", "||", s2)
    return s2

def _apply_domain_rules(s: str) -> str:
    # Wi-Fi SSID/PASS 패턴 교정 예시
    s = re.sub(r"\bP[Aa][Ss][Ss]\b", "PASS", s)
    s = re.sub(r"\b[S5]{2}[Ii1][Dd]\b", "SSID", s)
    return s

def _postprocess(text, boxes):
    text = _fix_confusions(text)
    text = _apply_domain_rules(text)
    new_boxes = []
    for b in boxes:
        t = _apply_domain_rules(_fix_confusions(b["text"] or ""))
        new_boxes.append({**b, "text": t})
    return text, new_boxes


# ---------------------------
# 셀(조각) 리스트 OCR
# ---------------------------
def run_ocr_on_list(images_bgr):
    texts, all_boxes = [], []
    for im in images_bgr:
        t, b = run_ocr(im)
        if t:
            texts.append(t)
        all_boxes.extend(b)
    return " ".join(texts).strip(), all_boxes
