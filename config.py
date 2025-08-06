# config.py
import os

# 기본 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 모델 및 데이터 경로 설정
EMBEDDING_MODEL = 'upskyy/bge-m3-korean'
LLM_MODEL = 'gemini-1.5-pro-latest'

VECTOR_STORE_DIR = os.path.join(BASE_DIR, 'vector_store')
ARTIFACT_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, 'artifacts.index')
ARTIFACT_DF_PATH = os.path.join(VECTOR_STORE_DIR, 'artifacts_df.pkl')
HISTORY_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, 'history.index')
HISTORY_DF_PATH = os.path.join(VECTOR_STORE_DIR, 'history_df.pkl')