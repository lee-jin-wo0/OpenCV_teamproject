from pydantic import BaseModel
from typing import List, Optional

class OCRBox(BaseModel):
    text: str
    conf: float
    box: List[List[float]]  # 4점 좌표 [[x1,y1],...]

class ProcessResponse(BaseModel):
    merged_used: bool
    text: str
    boxes: List[OCRBox]
    processed_image_b64: str  # Data URL (image/png)
