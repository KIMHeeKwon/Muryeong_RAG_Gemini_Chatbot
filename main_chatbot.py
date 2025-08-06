import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGChatbot:
    """
    미리 구축된 벡터 스토어를 기반으로 질의응답을 수행하는 RAG 챗봇 클래스.
    """
    def __init__(self, model_name='upskyy/bge-m3-korean'): # 모델 이름 업데이트
        print("⏳ 챗봇 초기화 시작...")
        
        # 1. 임베딩 모델 로드
        print(f"  - 임베딩 모델({model_name}) 로드 중...")
        self.model = SentenceTransformer(model_name)
        print("  - 임베딩 모델 로드 완료.")
        
        # 2. 벡터 스토어 및 데이터 로드
        self.artifact_index, self.artifact_df = self._load_vector_store('artifacts')
        self.history_index, self.history_df = self._load_vector_store('history')
        
        print("✅ 챗봇 초기화 완료. 질문을 입력할 준비가 되었습니다.")

    def _load_vector_store(self, store_name: str):
        """FAISS 인덱스와 데이터프레임을 로드하는 내부 함수."""
        index_path = os.path.join('vector_store', f'{store_name}.index')
        df_path = os.path.join('vector_store', f'{store_name}_df.pkl')
        
        if not os.path.exists(index_path) or not os.path.exists(df_path):
            print(f"🚨 치명적 오류: '{store_name}'의 벡터 스토어 파일이 없습니다.")
            print("    vector_store_builder.py를 성공적으로 실행했는지 확인해주세요.")
            exit()
            
        index = faiss.read_index(index_path)
        with open(df_path, 'rb') as f:
            df = pickle.load(f)
            
        print(f"  - '{store_name}' 벡터 스토어 로드 완료 (인덱스 벡터 수: {index.ntotal}, 데이터 행 수: {len(df)})")
        return index, df

    def _route_query(self, query: str):
        """질문이 특정 유물을 지칭하는지, 아니면 일반 역사 질문인지를 분류합니다."""
        for name in self.artifact_df['명칭'].tolist():
            if name in query:
                print(f"  🔍 라우터: '{name}' 키워드 발견 -> '유물 정보'로 분류")
                return "artifact"
        
        print("  🔍 라우터: 특정 유물 키워드 없음 -> '역사 정보'로 분류")
        return "history"

    def _search(self, query: str, search_type: str, k: int = 3):
        """질문 유형에 따라 적절한 벡터 스토어에서 검색을 수행합니다."""
        query_embedding = self.model.encode([query])
        
        if search_type == "artifact":
            index = self.artifact_index
            df = self.artifact_df
            text_column = 'rag_document'
            k = 1 
        else: # history
            index = self.history_index
            df = self.history_df
            text_column = 'text_chunk'

        distances, indices = index.search(query_embedding, k)
        
        # 검색된 컨텍스트와 함께 메타데이터(URL, 이미지 URL 등)도 함께 반환
        retrieved_docs = []
        print("\n  - 검색된 정보 조각(Context):")
        for i, idx in enumerate(indices[0]):
            doc_info = df.iloc[idx].to_dict()
            doc_info['similarity'] = distances[0][i]
            retrieved_docs.append(doc_info)
            print(f"    {i+1}. (유사도: {doc_info['similarity']:.4f}) {doc_info[text_column][:100]}...")
            
        return retrieved_docs

    def ask(self, query: str):
        """
        사용자 질문에 대한 전체 RAG 파이프라인을 실행하고 최종 답변을 생성합니다.
        """
        print(f"\n" + "="*50)
        print(f"👤 사용자 질문: {query}")

        query_type = self._route_query(query)
        retrieved_docs = self._search(query, query_type)
        
        # LLM에 전달할 컨텍스트 텍스트 생성
        context_for_llm = ""
        for doc in retrieved_docs:
            source_file = doc.get('source_file', '유물 DB')
            context_for_llm += f"[출처: {source_file}]\n{doc.get('rag_document') or doc.get('text_chunk')}\n\n"

        prompt = f"""당신은 무령왕릉 전문 AI 도슨트입니다.
당신의 임무는 반드시 아래 [참고 자료]에만 근거하여 사용자의 [질문]에 대해 답변하는 것입니다.
자료에 없는 내용은 절대로 지어내지 말고, "자료에 없는 내용이라 답변할 수 없습니다."라고 솔직하게 답변하세요.

---
[참고 자료]
{context_for_llm.strip()}
---

[질문]
{query}

[답변]
"""
        
        print("\n" + "="*50)
        print("🤖 최종 생성 프롬프트 (이 내용을 기반으로 LLM이 답변을 생성합니다):")
        print(prompt)
        
        # 실제 LLM API 호출 대신, 여기서는 생성된 프롬프트와 참조 데이터를 반환합니다.
        # 이 정보를 사용하여 실제 웹 앱에서는 답변과 함께 이미지, URL 등을 표시할 수 있습니다.
        final_result = {
            "prompt": prompt,
            "retrieved_docs": retrieved_docs
        }
        
        print("✅ RAG 파이프라인 실행 완료.")
        return final_result

# --- 메인 실행 로직 ---
if __name__ == '__main__':
    chatbot = RAGChatbot()
    
    # --- 테스트 질문 ---
    result1 = chatbot.ask("왕비의 은팔찌는 어떤 모양이야?")
    # 답변과 함께 이미지 URL을 표시하는 방법 (웹 앱에서의 활용 예시)
    if result1['retrieved_docs'] and 'image_url' in result1['retrieved_docs'][0]:
        print(f"\n🖼️ 관련 이미지 URL: {result1['retrieved_docs'][0]['image_url']}")

    result2 = chatbot.ask("무령왕릉 지석의 글씨체는 어떤 특징을 가지고 있나요?")