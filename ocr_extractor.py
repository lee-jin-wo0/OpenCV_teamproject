import cv2
import numpy as np
import re
# import pytesseract # No longer needed
from PIL import Image
from image_utils import imclearborder, bwareaopen
from paddleocr import PaddleOCR

# Tesseract OCR 경로 설정 (Windows 사용자의 경우) - 이제 사용하지 않음
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# PaddleOCR 초기화 (한 번만 수행)
# lang='en' for English, lang='ko' for Korean
ocr = PaddleOCR(use_angle_cls=True, 
                det_model_dir='./paddleocr_models/det', 
                rec_model_dir='./paddleocr_models/rec_korean', 
                cls_model_dir='./paddleocr_models/cls', 
                show_log=False) 

def extract_card_info(image, title='OCR Preprocessed Image'):
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 노이즈 제거 (Median Blur)
        gray_image = cv2.medianBlur(gray_image, 3) # 커널 크기는 조절 가능

        binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)
        
        # 모폴로지 연산 (Dilate, Erode) - 글자 연결 및 노이즈 제거
        kernel = np.ones((2,2),np.uint8) # 커널 크기는 조절 가능
        binary_image = cv2.dilate(binary_image, kernel, iterations = 1)
        binary_image = cv2.erode(binary_image, kernel, iterations = 1)

        binary_image = bwareaopen(binary_image, 120)
        binary_image = imclearborder(binary_image, 5)
        
        # plt.figure()
        # plt.imshow(binary_image, cmap='gray')
        # plt.title(title)
        # plt.axis('off')
        # plt.show()

        # PaddleOCR를 사용하여 텍스트 추출
        # PaddleOCR은 파일 경로를 받거나 numpy 배열을 직접 받을 수 있음
        # 여기서는 numpy 배열을 PIL Image로 변환 후 다시 numpy 배열로 변환하여 전달
        # 또는 cv2.imwrite로 임시 파일을 저장 후 경로 전달
        # 간단하게 numpy 배열을 직접 전달하는 방법 사용
        result = ocr.ocr(binary_image, cls=True)
        
        text = ""
        if result and result[0]:
            for line in result[0]:
                text += line[1][0] + " " # 텍스트와 신뢰도 중 텍스트만 추출
        
        cleaned_text = text.strip()
        # print(f"\n--- OCR 결과 ({title}) ---")
        # print(cleaned_text)
        # print("--------------------")

        # --- 더 견고한 카드 종류 식별 로직 ---
        card_type = "Unknown"
        info = {'type': card_type, 'number': None, 'issue_date': None, 'raw_text': cleaned_text}

        # 1. 번호 형식으로 먼저 식별
        id_pattern = re.compile(r'(\d{6}\s*-\s*\d{7})')
        license_pattern = re.compile(r'(\d{2}-\d{2}-\d{6}-\d{2})')
        
        id_match = id_pattern.search(cleaned_text)
        license_match = license_pattern.search(cleaned_text)

        if id_match:
            card_type = "ID Card"
            info['number'] = id_match.group(0)
        elif license_match:
            card_type = "Driver's License"
            info['number'] = license_match.group(0)
        # 2. 번호 형식이 없다면, 키워드로 식별
        elif "주민등록증" in cleaned_text:
            card_type = "ID Card"
        elif "운전면허증" in cleaned_text:
            card_type = "Driver's License"
        
        info['type'] = card_type

        # 3. 발급일 추출
        date_pattern = re.compile(r'(\d{4}[ .]\d{2}[ .]\d{2})')
        date_match = date_pattern.search(cleaned_text)
        if date_match:
            info['issue_date'] = date_match.group(0)

        return info

    except Exception as e:
        print(f"OCR 처리 중 오류 발생: {e}")
        return {'type': 'Error', 'number': None, 'issue_date': None, 'raw_text': ''}
