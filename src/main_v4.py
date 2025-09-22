# -*- coding: utf-8 -*-
"""
main_v4.py
- 입력:  단일 이미지 (--input)
- 출력:  1) 반사광 제거(+CLAHE) 결과: <output_basename>_glare.jpg
         2) 문서 윤곽이 충분히 크면 정투영 결과: --output 경로
            충분하지 않으면 정투영 생략하고 glare 결과를 --output 으로 저장
- OCR:   결과 이미지에 대해 PaddleOCR 수행(설치/호환 OK인 경우). 텍스트는 터미널 출력만.
"""

import os
import argparse
from typing import Optional, Tuple

import cv2
import numpy as np

# ===========================
#  OCR (선택적 사용) 준비
# ===========================
HAS_PADDLE = True
PADDLE_IMPORT_ERR = None

# NumPy 2.0+ 는 imgaug(=PaddleOCR 내부 의존)와 충돌 → 친절 메시지 후 OCR 비활성화
try:
    _np_ver_tuple = tuple(int(x) for x in np.__version__.split(".")[:2])
    if _np_ver_tuple >= (2, 0):
        HAS_PADDLE = False
        PADDLE_IMPORT_ERR = (
            f"현재 NumPy {np.__version__} 이므로 imgaug와 충돌합니다. "
            "OCR을 사용하려면 가상환경에서 다음을 실행하세요:\n"
            "  python -m pip install --upgrade 'pip<24'\n"
            "  python -m pip install --no-cache-dir numpy==1.26.4 imgaug==0.4.0\n"
            "그 후 다시 실행해 주세요."
        )
except Exception:
    pass

if HAS_PADDLE:
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception as e:
        HAS_PADDLE = False
        PADDLE_IMPORT_ERR = f"PaddleOCR import 실패: {e}\n→ OCR 단계는 건너뜁니다."

if not HAS_PADDLE and PADDLE_IMPORT_ERR:
    print(f"[경고] {PADDLE_IMPORT_ERR}")


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


# ===========================
#  글레어(반사광) 처리
# ===========================
def find_glare_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    HSV에서 밝고(S 낮음, V 높음) 영역을 글레어로 판단.
    주민등록증처럼 밝은 배경을 고려해 임계값을 약간 타이트하게 설정.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # 필요 시 아래 범위를 조정하세요.
    lower = np.array([0, 0, 200], dtype=np.uint8)     # V 높게
    upper = np.array([179, 90, 255], dtype=np.uint8)  # S 낮음~중간
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask  # 0/255


def remove_glare_inpaint(image_bgr: np.ndarray) -> np.ndarray:
    mask = find_glare_mask(image_bgr)

    # inpaint
    inpainted = cv2.inpaint(image_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # CLAHE (L 채널만)
    lab = cv2.cvtColor(inpainted, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    out = cv2.merge((l2, a, b))
    out_bgr = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
    return out_bgr


# ===========================
#  문서 경계 검출 & 정투영
# ===========================
def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _quad_area(pts: np.ndarray) -> float:
    # 사각형을 두 삼각형으로 나눠 근사 면적 계산
    rect = order_points(pts)
    tl, tr, br, bl = rect
    tri1 = 0.5 * abs(np.cross(tr - tl, bl - tl))
    tri2 = 0.5 * abs(np.cross(tr - br, bl - br))
    return float(tri1 + tri2)


def find_document_quad(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 1) Canny 기반
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:12]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    # 2) 적응형 이진화 기반
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 10
    )
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:12]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    return None


def warp_document(image_bgr: np.ndarray, quad4: np.ndarray) -> np.ndarray:
    rect = order_points(quad4)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxW = int(max(widthA, widthB))
    maxH = int(max(heightA, heightB))
    maxW = max(maxW, 1)
    maxH = max(maxH, 1)

    dst = np.array(
        [[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32"
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image_bgr, M, (maxW, maxH))
    return warped


# ===========================
#  OCR
# ===========================
def _init_ocr_instance(lang: str):
    """
    PaddleOCR 인스턴스 생성 시 자주 터지는 케이스를 포괄적으로 잡아
    사용자에게 다음 조치 안내만 하고 None 반환.
    """
    if not HAS_PADDLE:
        return None
    try:
        # 구버전/신버전 모두 커버
        try:
            return PaddleOCR(lang=lang, use_angle_cls=False)
        except TypeError:
            return PaddleOCR(lang=lang)
    except Exception as e:
        print(f"[경고] PaddleOCR 초기화 실패({lang}): {e}")
        print(
            "→ 조치:\n"
            "  1) 모델 캐시 삭제:  Remove-Item -Recurse -Force $env:USERPROFILE\\.paddleocr\n"
            "  2) (권장) 캐시 경로를 ASCII로:  $env:PADDLEOCR_HOME='C:\\paddleocr_cache'\n"
            "  3) 의존성 확인:  python -m pip install --no-cache-dir requests shapely pyclipper lmdb imgaug==0.4.0\n"
            "  4) NumPy는 1.26.4 권장:  python -m pip install --no-cache-dir numpy==1.26.4\n"
        )
        return None


def do_ocr(img_bgr: np.ndarray) -> None:
    if not HAS_PADDLE:
        print("PaddleOCR 미설치/비활성(호환성 문제)로 OCR 건너뜀")
        return

    ocr_ko = _init_ocr_instance("korean")
    ocr_en = _init_ocr_instance("en")

    if ocr_ko is None and ocr_en is None:
        print("(OCR 인스턴스 없음) → OCR 생략")
        return

    if ocr_ko is not None:
        print("\n--- OCR (KO) ---")
        try:
            res_ko = ocr_ko.ocr(img_bgr, cls=False)
            if res_ko and len(res_ko) > 0:
                for line in res_ko[0]:
                    txt, conf = line[1][0], line[1][1]
                    print(f"{txt} (conf: {conf:.2f})")
            else:
                print("(인식 없음)")
        except Exception as e:
            print(f"[경고] KO OCR 실패: {e}")

    if ocr_en is not None:
        print("\n--- OCR (EN) ---")
        try:
            res_en = ocr_en.ocr(img_bgr, cls=False)
            if res_en and len(res_en) > 0:
                for line in res_en[0]:
                    txt, conf = line[1][0], line[1][1]
                    print(f"{txt} (conf: {conf:.2f})")
            else:
                print("(인식 없음)")
        except Exception as e:
            print(f"[경고] EN OCR 실패: {e}")


# ===========================
#  파이프라인
# ===========================
def process_one_image(
    input_path: str,
    output_path: str,
    area_threshold: float,
    aspect_min: float,
    aspect_max: float,
) -> Tuple[bool, Optional[str]]:
    if not os.path.exists(input_path):
        return False, f"[에러] 입력 이미지가 없습니다: {input_path}"

    img = cv2.imread(input_path)
    if img is None:
        return False, f"[에러] 이미지를 읽을 수 없습니다: {input_path}"

    H, W = img.shape[:2]

    # 1) 반사광 제거 + 대비 향상
    glare_fixed = remove_glare_inpaint(img)

    # 보조 저장(디버깅용)
    base, _ = os.path.splitext(output_path)
    glare_only_path = f"{base}_glare.jpg"
    ensure_parent_dir(glare_only_path)
    cv2.imwrite(glare_only_path, glare_fixed)

    # 2) 문서 경계 → 정투영 (면적/비율 필터)
    quad = find_document_quad(glare_fixed)
    warped = None
    if quad is not None:
        quad_area = _quad_area(quad)
        area_ratio = quad_area / float(W * H)

        rect = order_points(quad)
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        w_est = max(widthA, widthB) + 1e-6
        h_est = max(heightA, heightB) + 1e-6
        aspect = w_est / h_est

        if area_ratio >= area_threshold and (aspect_min <= aspect <= aspect_max):
            warped = warp_document(glare_fixed, quad)
        else:
            print(f"[정보] 윤곽 필터로 정투영 스킵: area={area_ratio:.3f}, aspect={aspect:.2f}")
    else:
        print("[정보] 문서 경계 미검출 → 정투영 생략")

    # 3) 최종 저장
    ensure_parent_dir(output_path)
    final_img = warped if warped is not None else glare_fixed
    cv2.imwrite(output_path, final_img)
    print(f"✅ 결과 이미지 저장: {output_path}")

    # 4) OCR (터미널 출력)
    do_ocr(final_img)

    return True, None


def main():
    parser = argparse.ArgumentParser(description="반사광 제거 + 정투영 + OCR(터미널 출력)")
    parser.add_argument("--input", required=True, help="예: images/identification_card.jpg")
    parser.add_argument("--output", required=True, help="예: results/identification_card_result.jpg")
    parser.add_argument(
        "--area-threshold",
        type=float,
        default=0.005,  # 0.5% (주민등록증처럼 작게 찍힌 경우도 정투영 시도)
        help="정투영을 시도하기 위한 문서 윤곽 면적 비율(0~1), 기본 0.005(=0.5%)",
    )
    parser.add_argument("--aspect-min", type=float, default=0.5, help="허용 최소 가로세로비")
    parser.add_argument("--aspect-max", type=float, default=3.0, help="허용 최대 가로세로비")
    args = parser.parse_args()

    print("✨ 無色無光: 반사광 제거 문서 스캐너 – 시작!")
    ok, err = process_one_image(
        args.input,
        args.output,
        area_threshold=args.area_threshold,
        aspect_min=args.aspect_min,
        aspect_max=args.aspect_max,
    )
    if not ok:
        print(err)


if __name__ == "__main__":
    main()
