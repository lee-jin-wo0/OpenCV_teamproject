import base64
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel
from PIL import Image
import numpy as np
import cv2

from schemas import ProcessResponse, OCRBox
from processing import pipeline
from ocr import run_ocr

app = FastAPI(title="Glare OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

def bgr_to_png_b64(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{data}"

@app.post("/process", response_model=ProcessResponse)
async def process_images(files: List[UploadFile] = File(...)):
    # 파일 바이트 수집
    bytes_list = [await f.read() for f in files]

    processed_bgr, merged_used = pipeline(bytes_list)
    text, boxes = run_ocr(processed_bgr)

    return ProcessResponse(
        merged_used=merged_used,
        text=text,
        boxes=[OCRBox(**b) for b in boxes],
        processed_image_b64=bgr_to_png_b64(processed_bgr)
    )
