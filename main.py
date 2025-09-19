import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import re

from homomorphic_filter_module import homomorphic_filter_yuv
from ocr_extractor import extract_card_info

if __name__ == '__main__':
    # --- 분석할 이미지 경로를 여기에 지정하세요 ---
    image_path = 'image/image.png' 
    
    output_dir = 'image/filtered_images'
    
    try:
        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if original_image is None:
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

        # 감마 값 범위 설정
        gamma1_values = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        gamma2_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

        best_info = {'score': -1, 'gamma1': None, 'gamma2': None, 'extracted_data': None}

        for g1 in gamma1_values:
            for g2 in gamma2_values:
                print(f"\n--- 감마값 시도: gamma1={g1}, gamma2={g2} ---")
                filtered_image = homomorphic_filter_yuv(original_image, sigma=10, gamma1=g1, gamma2=g2)
                extracted_info = extract_card_info(filtered_image, title=f'Cleaned OCR Image (g1={g1}, g2={g2})')

                # OCR 결과 평가 (더 많은 정보가 추출될수록 높은 점수)
                current_score = 0
                if extracted_info.get('type') != 'Unknown':
                    current_score += 3
                if extracted_info.get('number'):
                    current_score += 2
                if extracted_info.get('issue_date'):
                    current_score += 1
                
                # raw_text의 길이를 점수에 추가 (더 많은 텍스트가 인식될수록 좋다고 가정)
                if extracted_info.get('raw_text'):
                    current_score += len(extracted_info['raw_text']) / 100 # 텍스트 길이에 비례하여 점수 추가

                if current_score > best_info['score']:
                    best_info['score'] = current_score
                    best_info['gamma1'] = g1
                    best_info['gamma2'] = g2
                    best_info['extracted_data'] = extracted_info
                
                # 중간 결과 출력 (선택 사항)
                print(f"현재 감마값 (g1={g1}, g2={g2}) 결과: {extracted_info.get('type', 'Unknown')}, {extracted_info.get('number', 'None')}, {extracted_info.get('issue_date', 'None')}")

        # 원본 이미지 파일 이름 추출
        original_image_name = os.path.basename(image_path)
        name_without_ext = os.path.splitext(original_image_name)[0]

        # 최적의 감마값으로 필터링된 이미지 저장 (조건부)
        if best_info['extracted_data'] and \
           (best_info['extracted_data'].get('type') != 'Unknown' or \
            best_info['extracted_data'].get('number') or \
            best_info['extracted_data'].get('issue_date')):
            
            final_filtered_image = homomorphic_filter_yuv(original_image, sigma=10, gamma1=best_info['gamma1'], gamma2=best_info['gamma2'])
            output_filename = f"filtered_{name_without_ext}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, final_filtered_image)
            print(f"최적 감마값으로 필터링된 이미지가 '{output_path}' 경로에 저장되었습니다.")
        else:
            print("유의미한 정보를 추출하지 못하여 필터링된 이미지를 저장하지 않습니다.")

        final_filtered_image = None
        # 최적의 감마값으로 필터링된 이미지 저장 (조건부)
        if best_info['extracted_data'] and \
           (best_info['extracted_data'].get('type') != 'Unknown' or \
            best_info['extracted_data'].get('number') or \
            best_info['extracted_data'].get('issue_date')):
            
            final_filtered_image = homomorphic_filter_yuv(original_image, sigma=10, gamma1=best_info['gamma1'], gamma2=best_info['gamma2'])
            output_filename = f"filtered_{name_without_ext}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, final_filtered_image)
            print(f"최적 감마값으로 필터링된 이미지가 '{output_path}' 경로에 저장되었습니다.")
        else:
            print("유의미한 정보를 추출하지 못하여 필터링된 이미지를 저장하지 않습니다.")

        # 최적의 감마값을 가진 사진을 화면에 띄우기
        if best_info['gamma1'] is not None:
            if final_filtered_image is None: # 이미 저장된 경우 다시 필터링할 필요 없음
                final_filtered_image = homomorphic_filter_yuv(original_image, sigma=10, gamma1=best_info['gamma1'], gamma2=best_info['gamma2'])
            
            plt.figure(figsize=(8, 6))
            plt.imshow(cv2.cvtColor(final_filtered_image, cv2.COLOR_BGR2RGB))
            plt.title(f'Best Homomorphic Filtered Image (g1={best_info["gamma1"]}, g2={best_info["gamma2"]})')
            plt.axis('off')
            plt.show()

        # 최종 결과 출력
        print("\n--- 최종 추출된 정보 (최적 감마값) ---")
        if best_info['extracted_data']:
            print(f"최적 감마1: {best_info['gamma1']}, 최적 감마2: {best_info['gamma2']}")
            print(f"카드 종류: {best_info['extracted_data'].get('type', '알 수 없음')}")
            
            if best_info['extracted_data'].get('number'):
                print(f"번호: {best_info['extracted_data']['number']}")
            else:
                print("번호를 찾지 못했습니다.")
            
            if best_info['extracted_data'].get('issue_date'):
                cleaned_date = re.sub(r'[ .]', '.', best_info['extracted_data']['issue_date'])
                print(f"발급일: {cleaned_date}")
            else:
                print("발급일을 찾지 못했습니다.")
            print(f"인식된 전체 텍스트:\n{best_info['extracted_data'].get('raw_text', '')}")
        else:
            print("어떤 감마값으로도 정보를 추출하지 못했습니다.")
        print("------------------------")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")