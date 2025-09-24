# from typing import List
# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import base64
# import cv2
# import numpy as np

# from processing import pipeline, find_grid_cells_3x3
# from ocr import run_ocr, run_ocr_on_list


# # ---------------------------
# # Pydantic 스키마
# # ---------------------------
# class OCRBox(BaseModel):
#     text: str
#     conf: float
#     box: List[List[float]]  # [[x,y], ... 4점]

# class ProcessResponse(BaseModel):
#     merged_used: bool
#     text: str
#     boxes: List[OCRBox]
#     processed_image_b64: str  # data URL 포함


# # ---------------------------
# # 유틸
# # ---------------------------
# def bgr_to_png_b64(bgr: np.ndarray) -> str:
#     if bgr is None or getattr(bgr, "size", 0) == 0:
#         return ""
#     # 인코딩 안전화
#     bgr = np.clip(bgr, 0, 255).astype(np.uint8)
#     bgr = np.ascontiguousarray(bgr)

#     ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
#     mime = "image/png"
#     if not ok:
#         ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
#         mime = "image/jpeg"
#     if not ok:
#         return ""

#     b64 = base64.b64encode(buf.tobytes()).decode("ascii")
#     # ⬇️ 프리픽스 포함
#     return f"data:{mime};base64,{b64}"


# # ---------------------------
# # FastAPI
# # ---------------------------
# app = FastAPI(title="Glare OCR API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
# )


# @app.get("/health")
# def health():
#     return {"ok": True}


# @app.post("/warmup")
# def warmup():
#     # PaddleOCR 초기 로딩 워밍업
#     img = np.zeros((64, 64, 3), dtype=np.uint8)
#     _ = run_ocr(img)
#     return {"warmed": True}


# @app.post("/process", response_model=ProcessResponse)
# async def process_images(files: List[UploadFile] = File(...)):
#     # 1) 업로드 읽기
#     bytes_list = [await f.read() for f in files]

#     # 2) 전처리(화이트보드 특화 반사 제거 + 시점보정 + 품질 부스팅)
#     processed_bgr, merged_used = pipeline(bytes_list)

#     # 3) 3x3 격자(있으면) → 셀별 OCR, 아니면 전체 OCR
#     try:
#         cells = find_grid_cells_3x3(processed_bgr)
#     except Exception:
#         cells = []

#     try:
#         if cells:
#             text, boxes = run_ocr_on_list(cells)
#         else:
#             text, boxes = run_ocr(processed_bgr)
#     except Exception as e:
#         print("OCR error:", e)
#         text, boxes = "", []

#     # 4) 응답
#     return ProcessResponse(
#         merged_used=merged_used,
#         text=text,
#         boxes=[OCRBox(**b) for b in boxes],
#         processed_image_b64=bgr_to_png_b64(processed_bgr),
#     )


from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2
import numpy as np

from processing import pipeline, find_grid_cells_3x3
from ocr import run_ocr, run_ocr_on_list

# ---------------------------
# Pydantic 스키마
# ---------------------------
class OCRBox(BaseModel):
    text: str
    conf: float
    box: List[List[float]]  # [[x,y], ... 4점]

class ProcessResponse(BaseModel):
    merged_used: bool
    text: str
    boxes: List[OCRBox]
    processed_image_b64: str  # data URL 포함
    # ✅ 추가: 품질 가시화
    mean_conf: float = 0.0
    variant: str = "raw"

# ---------------------------
# 유틸
# ---------------------------
def bgr_to_png_b64(bgr: np.ndarray) -> str:
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return ""
    bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    bgr = np.ascontiguousarray(bgr)

    ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    mime = "image/png"
    if not ok:
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        mime = "image/jpeg"
    if not ok:
        return ""

    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"

# ---------------------------
# FastAPI
# ---------------------------
app = FastAPI(title="Glare OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/warmup")
def warmup():
    # PaddleOCR 초기 로딩 워밍업
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    _ = run_ocr(img)
    return {"warmed": True}

@app.post("/process", response_model=ProcessResponse)
async def process_images(files: List[UploadFile] = File(...)):
    # 1) 업로드 읽기
    bytes_list = [await f.read() for f in files]

    # 2) 전처리(글레어 제거/조명 평탄화/시점 보정 등)
    processed_bgr, merged_used = pipeline(bytes_list)

    # 3) 3x3 격자 → 셀별 OCR, 아니면 전체 OCR
    try:
        cells = find_grid_cells_3x3(processed_bgr)
    except Exception:
        cells = []

    mean_conf = 0.0
    variant = "raw"
    try:
        if cells:
            res = run_ocr_on_list(cells)
        else:
            res = run_ocr(processed_bgr)

        # ✅ 새/옛 반환 호환: (text, boxes) | (text, boxes, mean_conf, variant)
        if isinstance(res, tuple) and len(res) >= 2:
            text, boxes = res[0], res[1]
            if len(res) >= 3: mean_conf = float(res[2])
            if len(res) >= 4: variant = str(res[3])
        else:
            text, boxes = "", []
    except Exception as e:
        print("OCR error:", e)
        text, boxes = "", []

    # 4) 응답
    return ProcessResponse(
        merged_used=merged_used,
        text=text,
        boxes=[OCRBox(**b) for b in boxes],
        processed_image_b64=bgr_to_png_b64(processed_bgr),
        mean_conf=mean_conf,
        variant=variant,
    )
