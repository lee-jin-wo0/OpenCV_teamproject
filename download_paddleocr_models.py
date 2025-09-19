from paddleocr import PaddleOCR

# Initialize PaddleOCR for Korean to trigger model download
# This will download the models to ~/.paddleocr/ if they are not already present
try:
    ocr = PaddleOCR(use_angle_cls=True, lang='ko', show_log=True) # show_log=True to see download progress
    print("PaddleOCR Korean models initialized successfully (or already present).")
except Exception as e:
    print(f"Error initializing PaddleOCR: {e}")
