# data_preprocessor.py
import pandas as pd
import re
import os

def preprocess_artifacts_csv(input_path: str, output_path: str, base_url: str):
    """
    (이미지 파일명 오류 수정 버전)
    '소장품번호'와 '세부번호'를 0으로 채워(padding) 정확한 이미지 경로를 생성합니다.
    """
    print(f"🔄 CSV 정제 프로세스 시작: '{input_path}'")
    
    try:
        df = pd.read_csv(input_path)
        
        essential_columns = [
            'id', '명칭', '소장품번호', '세부번호', '국적/시대1', '재질1', 
            '지정구분', '특징', '신보고서 종합편 설명 내용', 
            'MUCH_URL', '참고자료'
        ]
        df_processed = df[essential_columns].copy()

        for col in essential_columns:
            df_processed[col] = df_processed[col].astype(str).fillna('')
        
        for col in ['MUCH_URL', 'id', '소장품번호', '세부번호']:
             df_processed[col] = df_processed[col].str.strip()
        
        # (⭐ 핵심 수정) '소장품번호'와 '세부번호'를 0으로 채워(padding) URL 생성
        def create_image_url_from_ids(row):
            try:
                main_no_str = str(row['소장품번호']).strip()
                sub_no_str = str(row['세부번호']).strip()

                if not main_no_str.isdigit() or not sub_no_str.isdigit():
                    return ""
                
                # 소장품번호를 6자리로 패딩 (예: 1 -> 000001)
                main_no_padded = f"{int(main_no_str):06d}"
                
                # 세부번호를 5자리로 패딩 (예: 0 -> 00000)
                sub_no_padded = f"{int(sub_no_str):05d}"
                
                # 패딩된 세부번호를 00-00 형식으로 변환
                sub_no_formatted = f"{sub_no_padded[:2]}-{sub_no_padded[2:4]}"

                # 최종 파일 이름 생성 (예: mur000001-00-00.jpg)
                file_name = f"mur{main_no_padded}-{sub_no_formatted}.jpg"
                return f"{base_url}/{file_name}"
            except (ValueError, TypeError, IndexError):
                return ""

        df_processed['image_url'] = df_processed.apply(create_image_url_from_ids, axis=1)
        print("  - '소장품번호'와 '세부번호' 기준으로 이미지 URL 재생성 완료.")

        def create_rag_document(row):
            return (
                f"[유물명]: {row['명칭']}\n[시대]: {row['국적/시대1']}\n"
                f"[재질]: {row['재질1']}\n[지정 정보]: {row['지정구분']}\n"
                f"[주요 특징]: {row['특징']}\n[상세 설명]: {row['신보고서 종합편 설명 내용']}\n"
                f"[참고 자료]: {row['참고자료']}"
            )
        df_processed['rag_document'] = df_processed.apply(create_rag_document, axis=1)
        df_processed['rag_document'] = df_processed['rag_document'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

        final_df = df_processed[['id', '명칭', '소장품번호', 'rag_document', 'MUCH_URL', 'image_url']]
        
        final_df = final_df.dropna(subset=['id'])
        final_df = final_df[final_df['id'] != 'nan']

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"✅ CSV 정제 완료: '{output_path}'에 가장 정확한 이미지 경로가 포함된 새 파일이 저장되었습니다.")
        
    except Exception as e:
        print(f"🚨 오류: 데이터 처리 중 문제 발생 - {e}")

if __name__ == '__main__':
    INPUT_CSV_PATH = os.path.join('data', 'converted_검증완료_통합.csv')
    OUTPUT_CSV_PATH = os.path.join('data', 'preprocessed_artifacts_final_with_images.csv')
    BASE_IMAGE_URL = "/static/images" 
    preprocess_artifacts_csv(INPUT_CSV_PATH, OUTPUT_CSV_PATH, BASE_IMAGE_URL)
