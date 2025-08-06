# vector_store_builder.py
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import os
import pickle

def build_and_save_vector_store(
    data_path: str, 
    text_column: str, 
    index_output_path: str,
    dataframe_output_path: str,
    model: SentenceTransformer
):
    """
    주어진 CSV 파일의 텍스트 데이터를 임베딩하고, FAISS 인덱스와 원본 데이터프레임을 저장합니다.
    """
    print(f"🔄 '{data_path}' 파일 처리 시작...")
    
    try:
        df = pd.read_csv(data_path)
        df[text_column] = df[text_column].fillna('')
        texts = df[text_column].tolist()
        
        if not texts:
            print(f"🚨 경고: '{data_path}'에 처리할 텍스트가 없습니다.")
            return

        print(f"  - 텍스트 데이터 로드 완료. 총 {len(texts)}개 항목 임베딩 중...")
        
        embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        embeddings_np = embeddings.cpu().numpy()
        
        print(f"  - 임베딩 완료. 벡터 차원: {embeddings_np.shape[1]}")
        
        index = faiss.IndexFlatL2(embeddings_np.shape[1])
        index.add(embeddings_np)
        
        print(f"  - FAISS 인덱스 구축 완료. 인덱스에 {index.ntotal}개 벡터 포함.")
        
        os.makedirs(os.path.dirname(index_output_path), exist_ok=True)
        faiss.write_index(index, index_output_path)
        
        with open(dataframe_output_path, 'wb') as f:
            pickle.dump(df, f)
            
        print(f"✅ 완료: 벡터 DB는 '{index_output_path}'에, 데이터는 '{dataframe_output_path}'에 저장되었습니다.")

    except FileNotFoundError:
        print(f"🚨 오류: 입력 파일 '{data_path}'을 찾을 수 없습니다.")
    except Exception as e:
        print(f"🚨 오류: 벡터 스토어 구축 중 예상치 못한 문제가 발생했습니다 - {e}")

if __name__ == '__main__':
    print("⏳ 임베딩 모델(upskyy/bge-m3-korean) 로딩 중...")
    embedding_model = SentenceTransformer('upskyy/bge-m3-korean')
    print("✅ 임베딩 모델 로드 완료!")

    # --- 유물 정보 벡터 DB 구축 ---
    build_and_save_vector_store(
        data_path=os.path.join('data', 'preprocessed_artifacts_final_with_images.csv'),
        text_column='rag_document',
        index_output_path=os.path.join('vector_store', 'artifacts.index'),
        dataframe_output_path=os.path.join('vector_store', 'artifacts_df.pkl'),
        model=embedding_model
    )
    
    print("-" * 50)

    # --- 역사 정보 벡터 DB 구축 ---
    build_and_save_vector_store(
        data_path=os.path.join('data', 'preprocessed_history_chunks_sectioned.csv'),
        text_column='text_chunk',
        index_output_path=os.path.join('vector_store', 'history.index'),
        dataframe_output_path=os.path.join('vector_store', 'history_df.pkl'),
        model=embedding_model
    )
