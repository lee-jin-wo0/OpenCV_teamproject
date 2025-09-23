import os
from paddleocr import PaddleOCR

# Ko + En
_ocr = PaddleOCR(use_angle_cls=True, lang='korean')  # 'korean' packs ko+en

def run_ocr(image_bgr):
    """
    Returns: text(str), boxes(list of dict with text/conf/box)
    """
    import cv2
    result = _ocr.ocr(image_bgr, cls=True)
    lines = []
    boxes = []
    for page in result:
        for box, (txt, conf) in page:
            lines.append(txt)
            boxes.append({
                "text": txt,
                "conf": float(conf),
                "box": [[float(x), float(y)] for x, y in box]
            })
    return "\n".join(lines), boxes
