import fitz  # PyMuPDF
import pandas as pd
import os
import re

def sectionize_and_preprocess_pdfs(pdf_directory: str, output_path: str, target_chunk_size: int = 1500, min_chunk_length: int = 100):
    """
    (전략 변경: Sectioning 버전)
    PDF 텍스트를 적절한 크기의 의미있는 '구획(Section)'으로 묶어 저장합니다.

    Args:
        pdf_directory (str): PDF 파일들이 있는 폴더 경로.
        output_path (str): 정제된 텍스트 구획을 저장할 CSV 파일 경로.
        target_chunk_size (int): 목표로 하는 구획의 글자 수.
        min_chunk_length (int): 유의미한 구획으로 간주할 최소 글자 수.
    """
    print("🔄 PDF 처리 프로세스 시작 (Sectioning 전략)...")
    
    all_sections = []
    
    try:
        # ... (이전과 동일한 파일 경로 및 존재 여부 확인 로직) ...
        if not os.path.isdir(pdf_directory):
            print(f"🚨 오류: '{pdf_directory}' 폴더를 찾을 수 없습니다.")
            return

        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
        print(f"  - 총 {len(pdf_files)}개의 PDF 파일을 발견했습니다: {pdf_files}")

        for filename in pdf_files:
            file_path = os.path.join(pdf_directory, filename)
            print(f"  - '{filename}' 파일 처리 중...")
            
            doc = fitz.open(file_path)
            full_text = "".join(page.get_text("text") + "\n" for page in doc)
            doc.close()

            # 1. 텍스트를 문단 단위로 분할
            paragraphs = full_text.split('\n\n')
            
            # 2. 문단을 합쳐 구획(Section) 생성
            current_section = ""
            processed_sections = 0
            for p in paragraphs:
                cleaned_p = re.sub(r'\s+', ' ', p).strip()
                if not cleaned_p:
                    continue

                # 현재 구획에 문단을 추가했을 때 목표 크기를 넘는지 확인
                if len(current_section) + len(cleaned_p) + 1 > target_chunk_size and len(current_section) > 0:
                    # 목표 크기를 넘으면, 현재까지의 구획을 저장
                    if len(current_section) >= min_chunk_length:
                        all_sections.append({'source_file': filename, 'text_chunk': current_section})
                        processed_sections += 1
                    # 현재 문단으로 새로운 구획 시작
                    current_section = cleaned_p
                else:
                    # 목표 크기를 넘지 않으면, 현재 구획에 문단을 계속 추가
                    if current_section:
                        current_section += "\n\n" + cleaned_p
                    else:
                        current_section = cleaned_p
            
            # 마지막 남은 구획 저장
            if len(current_section) >= min_chunk_length:
                all_sections.append({'source_file': filename, 'text_chunk': current_section})
                processed_sections += 1
            
            print(f"    -> 의미있는 구획(Section) {processed_sections}개 생성 완료.")

        df_sections = pd.DataFrame(all_sections)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_sections.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"✅ 프로세스 완료: 총 {len(df_sections)}개의 텍스트 구획이 '{output_path}'에 저장되었습니다.")

    except Exception as e:
        print(f"🚨 오류: PDF 처리 중 예상치 못한 문제가 발생했습니다 - {e}")

# --- 모듈 실행 ---
if __name__ == '__main__':
    PDF_SOURCE_DIRECTORY = os.path.join('data', 'pdf_data')
    OUTPUT_CHUNK_PATH = os.path.join('data', 'preprocessed_history_chunks_sectioned.csv')
    
    sectionize_and_preprocess_pdfs(PDF_SOURCE_DIRECTORY, OUTPUT_CHUNK_PATH)
    
    print("\n--- 최종 생성된 텍스트 구획(Section) 샘플 ---")
    try:
        sample_df = pd.read_csv(OUTPUT_CHUNK_PATH)
        pd.set_option('display.max_colwidth', 150)
        print(sample_df.head())
        # 생성된 구획들의 길이 분포 확인
        print("\n--- 구획별 글자 수 통계 ---")
        print(sample_df['text_chunk'].str.len().describe())
    except FileNotFoundError:
        print("결과 파일을 찾을 수 없습니다.")