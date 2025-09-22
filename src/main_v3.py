import cv2
import numpy as np
import os
import glob
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from paddleocr import PaddleOCR
from logging import getLogger

# PaddleOCR 초기화 로그 레벨 조정
logger = getLogger()
logger.setLevel('ERROR')

def find_glare_mask(image):
    """HSV 색 공간과 모폴로지 연산을 이용해 글레어 마스크를 생성합니다."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # V(value)가 높고 S(saturation)가 낮은 영역을 반사광으로 간주합니다.
    lower_glare = np.array([0, 0, 180], dtype=np.uint8)
    upper_glare = np.array([179, 70, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_glare, upper_glare)
    
    # 작은 노이즈 제거를 위해 모폴로지 연산 적용
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def align_images(image1, image2):
    """ORB 특징점을 이용해 이미지2를 이미지1에 정합합니다."""
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 충분한 매칭점이 있을 때만 호모그래피 계산
    if len(matches) > 10:
        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        aligned_image = cv2.warpPerspective(image2, M, (image1.shape[1], image1.shape[0]))
        return aligned_image
    else:
        return None

def find_document_border(image):
    """이미지에서 가장 큰 사각형 윤곽선을 찾아 반환합니다."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        
        if len(approx) == 4:
            return approx
    return None

def warp_image(image, pts):
    """주어진 4개 점을 기준으로 이미지를 정투영 변환합니다."""
    # 꼭짓점 순서 정렬 (좌상, 우상, 우하, 좌하)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    (tl, tr, br, bl) = rect
    
    # 변환 후 이미지의 너비와 높이 계산
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    
    # 변환 행렬 계산 및 적용
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def main(input_folder, output_folder):
    """
    메인 파이프라인 실행 함수.
    """
    print("✨ 無色無光: 반사광 제거 문서 스캐너 프로젝트 시작!")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = sorted(glob.glob(os.path.join(input_folder, 'test_ex_paddle.jpg')))
    if not image_paths:
        print("입력 폴더에 .jpg 이미지가 없습니다. 이미지를 추가해주세요.")
        return

    # 첫 번째 이미지를 기준 프레임으로 설정
    base_image = cv2.imread(image_paths[0])
    
    aligned_images = [base_image]
    # 프레임 정합
    for i in range(1, len(image_paths)):
        img = cv2.imread(image_paths[i])
        aligned = align_images(base_image, img)
        if aligned is not None:
            aligned_images.append(aligned)
        else:
            print(f"이미지 {os.path.basename(image_paths[i])} 정합 실패, 원본 사용.")
            aligned_images.append(img)
            
    # 멀티프레임 합성 (반사광 제거)
    final_image = np.zeros_like(base_image, dtype=np.float32)
    glare_masks = [find_glare_mask(img) for img in aligned_images]

    for i in range(base_image.shape[0]):
        for j in range(base_image.shape[1]):
            best_pixel = None
            best_glare_metric = float('inf')

            # 각 프레임의 픽셀을 비교하여 반사광이 가장 적은 픽셀 선택
            for k in range(len(aligned_images)):
                is_glare = glare_masks[k][i, j]
                if is_glare == 0:  # 반사광이 없는 픽셀
                    best_pixel = aligned_images[k][i, j]
                    break
            
            if best_pixel is None:
                # 모든 프레임에 반사광이 있다면, 중간값 픽셀을 선택
                pixels = [img[i, j] for img in aligned_images]
                best_pixel = np.median(pixels, axis=0)

            final_image[i, j] = best_pixel

    final_image = final_image.astype(np.uint8)

    # 후처리: CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    result_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    # 문서 경계 찾기
    document_border = find_document_border(result_image)
    if document_border is not None:
        # 문서 영역만 잘라내어 정투영 변환
        warped_image = warp_image(result_image, document_border.reshape(4, 2))
    else:
        warped_image = result_image
        print("경계를 찾지 못해 전체 이미지를 사용합니다.")

    # 결과 저장
    result_path = os.path.join(output_folder, 'test_ex_paddle_result.jpg')
    cv2.imwrite(result_path, result_image)
    print(f"✅ 최종 결과 이미지가 '{result_path}'에 저장되었습니다.")
    
    warped_path = os.path.join(output_folder, 'test_ex_paddle_warped_result.jpg')
    cv2.imwrite(warped_path, warped_image)
    print(f"✅ 정투영된 결과 이미지가 '{warped_path}'에 저장되었습니다.")

    # 평가 및 리포트
    # ----------------
    print("\n--- OCR 성능 평가 및 품질지표 계산 ---")
    base_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    warped_gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)

    # PSNR/SSIM 계산
    psnr_value = psnr(base_gray, warped_gray)
    ssim_value = ssim(base_gray, warped_gray, data_range=warped_gray.max() - warped_gray.min())
    print(f"PSNR (원본 대비): {psnr_value:.2f} dB")
    print(f"SSIM (원본 대비): {ssim_value:.4f}")

    # OCR 결과를 위한 전처리 강화 (PaddleOCR은 내부 전처리가 강력하여 필요 없음)
    
    # PaddleOCR 인스턴스 생성 및 OCR 실행
    try:
        # 한국어와 영어 모델 로드
        ocr_eng = PaddleOCR(lang='en')
        ocr_kor = PaddleOCR(lang='ko')
        
        # 원본 이미지 OCR
        result_base_eng = ocr_eng.ocr(base_image, cls=True)
        result_base_kor = ocr_kor.ocr(base_image, cls=True)
        
        # 정투영된 이미지 OCR
        result_warped_eng = ocr_eng.ocr(warped_image, cls=True)
        result_warped_kor = ocr_kor.ocr(warped_image, cls=True)

        original_text_eng = "\n".join([line[1][0] for res in result_base_eng for line in res])
        original_text_kor = "\n".join([line[1][0] for res in result_base_kor for line in res])
        final_text_eng = "\n".join([line[1][0] for res in result_warped_eng for line in res])
        final_text_kor = "\n".join([line[1][0] for res in result_warped_kor for line in res])

        print("\n원본 이미지 OCR 결과 (영어):\n", "-"*20, "\n", original_text_eng)
        print("\n원본 이미지 OCR 결과 (한국어):\n", "-"*20, "\n", original_text_kor)
        print("\n최종 결과 이미지 OCR 결과 (영어):\n", "-"*20, "\n", final_text_eng)
        print("\n최종 결과 이미지 OCR 결과 (한국어):\n", "-"*20, "\n", final_text_kor)

    except Exception as e:
        print(f"OCR 실행 중 오류가 발생했습니다: {e}")
        print("PaddleOCR이 올바르게 설치되었는지 확인하고, 'paddlepaddle' 및 'paddleocr' 라이브러리를 설치해주세요.")

if __name__ == "__main__":
    input_dir = 'images'
    output_dir = 'results'
    main(input_dir, output_dir)
