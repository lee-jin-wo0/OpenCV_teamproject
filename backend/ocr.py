import os
from paddleocr import PaddleOCR

# Ko + En
_ocr = PaddleOCR(
    use_angle_cls=True, 
    lang='korean', 
    use_gpu=False,
    det_limit_side_len=1280,  # 이미지 해상도에 맞게 값 조정
    rec_image_shape='3, 48, 384' # 입력 이미지 크기 조정
)

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
