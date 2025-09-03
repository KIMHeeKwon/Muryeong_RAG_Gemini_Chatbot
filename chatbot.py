# chatbot.py
import os
import faiss
import pickle
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import config
import numpy as np
import json
from dotenv import load_dotenv

class RAGChatbot:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(RAGChatbot, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'): return
        load_dotenv()
        print("⏳ 챗봇 초기화 시작...")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.artifact_index, self.artifact_df = self._load_vector_store('artifacts', config.ARTIFACT_INDEX_PATH, config.ARTIFACT_DF_PATH)
        self.history_index, self.history_df = self._load_vector_store('history', config.HISTORY_INDEX_PATH, config.HISTORY_DF_PATH)
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key: raise ValueError("API 키가 .env에 없습니다.")
            genai.configure(api_key=api_key)
            self.llm_model = genai.GenerativeModel(config.LLM_MODEL)
            print("  - Google Gemini 모델 로드 완료.")
        except Exception as e:
            self.llm_model = None
            print(f"  - 🚨 경고: Gemini 모델 로드 실패 - {e}")
        self._initialized = True
        print("✅ 챗봇 초기화 완료.")

    def _load_vector_store(self, store_name, index_path, df_path):
        if not os.path.exists(index_path) or not os.path.exists(df_path):
            raise FileNotFoundError(f"'{store_name}'의 벡터 스토어 파일이 없습니다.")
        print(f"  - '{store_name}' 벡터 스토어 로딩...")
        index = faiss.read_index(index_path)
        with open(df_path, 'rb') as f:
            df = pickle.load(f)
        return index, df

    def _rewrite_query_with_history(self, query: str, chat_history: list):
        if not chat_history: return query
        
        def _get_message_text(msg: dict) -> str:
            if "content" in msg: return str(msg["content"])
            parts = msg.get("parts")
            if isinstance(parts, list): return "".join(str(p) for p in parts)
            return str(parts) if parts is not None else ""

        formatted_history = "\n".join([f"사용자: {_get_message_text(msg)}" if msg.get('role') == 'user' else f"진묘: {_get_message_text(msg)}" for msg in chat_history])
        
        rewrite_prompt = f"""이전 대화 내용은 다음과 같습니다:
---
{formatted_history}
---
위 대화의 맥락을 고려하여, 사용자의 마지막 질문인 "{query}"를, 이전 대화 내용을 모르는 사람도 이해할 수 있는 완전하고 독립적인 질문으로 다시 작성해주세요.
재작성된 질문은 다른 어떤 설명도 없이, 오직 질문 문장 하나만 있어야 합니다.
"""
        try:
            response = self.llm_model.generate_content(rewrite_prompt)
            rewritten_query = response.text.strip().replace('"', '')
            print(f"  ✍️ 질문 재구성: '{query}' -> '{rewritten_query}'")
            return rewritten_query
        except Exception as e:
            print(f"🚨 질문 재구성 중 오류 발생: {e}")
            return query

    def _semantic_route_query(self, query: str):
        routing_prompt = f"""당신은 사용자의 질문 의도를 분석하는 라우팅 전문가입니다.
[사용자 질문]을 보고, 의도를 아래 [질문 유형] 중 하나로 분류하여 JSON 형식으로만 답변해주세요.
[중요 규칙]
- '무령왕릉' 자체의 발견, 역사, 배경 등에 대한 질문은 '역사_배경'으로 분류해야 합니다.
- '무령왕릉 지석', '무령왕릉 석수' 등 '무령왕릉' 뒤에 다른 단어가 붙는 경우는 '유물_상세정보'로 분류할 수 있습니다.
[질문 유형]
- "유물_상세정보": 특정 유물 하나에 대한 상세 정보(모양, 재질, 출토 위치 등)를 묻는 질문.
- "역사_배경": 특정 시대, 사건, 기술, 문화 등 포괄적인 역사적 배경이나 지식을 묻는 질문.
- "유물_비교": 두 개 이상의 유물을 비교해달라는 질문.
- "단순_대화": 정보 검색이 필요 없는 일반적인 대화 (인사, 감사 등).
[사용자 질문]
"{query}"
"""
        try:
            response = self.llm_model.generate_content(f'{routing_prompt}\n[분석 결과 (JSON 형식)]\n{{\n  "classification": "...",\n  "reason": "..."\n}}')
            json_text = response.text.strip().replace('```json', '').replace('```', '').strip()
            route_result = json.loads(json_text)
            classification = route_result.get("classification", "역사_배경")
            print(f"  🧠 시맨틱 라우터: '{classification}'으로 분류. (이유: {route_result.get('reason')})")
            return classification
        except Exception as e:
            print(f"🚨 라우팅 중 오류 발생: {e}. '역사_배경'으로 기본 설정합니다.")
            return "역사_배경"

    def _search(self, query: str, route: str, k: int = 3):
        query_embedding = self.model.encode([query])
        if route == "유물_상세정보":
            distances, indices = self.artifact_index.search(query_embedding, 1)
            return [self.artifact_df.iloc[idx].to_dict() for idx in indices[0]]
        elif route == "유물_비교":
            distances, indices = self.artifact_index.search(query_embedding, k)
            return [self.artifact_df.iloc[idx].to_dict() for idx in indices[0]]
        elif route == "역사_배경":
            distances, indices = self.history_index.search(query_embedding, k)
            return [self.history_df.iloc[idx].to_dict() for idx in indices[0]]
        else:
            return []

    def ask(self, query: str, chat_history: list = []):
        if not self.llm_model: return {"error": "Gemini 모델이 초기화되지 않았습니다."}
        
        # (⭐ 핵심 수정) 질문 재구성 단계에 '안전장치' 추가
        is_specific_query = any(name in query for name in self.artifact_df['명칭'].tolist())
        
        # 질문이 이미 명확한 유물 이름을 포함하고 있다면, 굳이 재구성하지 않음
        if is_specific_query and len(query) > 4:
             rewritten_query = query
             print(f"  ✍️ 질문이 명확하여 재구성 생략: '{query}'")
        else:
             rewritten_query = self._rewrite_query_with_history(query, chat_history)

        route = self._semantic_route_query(rewritten_query)
        
        if route == "단순_대화":
            try:
                response = self.llm_model.generate_content(f"사용자가 다음과 같이 말했습니다: '{query}'. 자세하고 친절하게 답변해주세요.")
                return {"answer": response.text, "metadata": []}
            except Exception as e:
                return {"error": f"Gemini API 호출 중 오류 발생: {e}"}

        retrieved_docs = self._search(rewritten_query, route)
        
        context_for_llm = ""
        for doc in retrieved_docs:
            source = doc.get('source_file', '유물 DB: ' + doc.get('명칭', ''))
            context_for_llm += f"### 참고 자료 (출처: {source}) ###\n"
            context_for_llm += f"내용: {doc.get('rag_document') or doc.get('text_chunk')}\n"
            if '소장품번호' in doc and doc['소장품번호']: context_for_llm += f"소장품번호: {doc['소장품번호']}\n"
            if 'MUCH_URL' in doc and doc['MUCH_URL']: context_for_llm += f"관련 링크: {doc['MUCH_URL']}\n"
            context_for_llm += "\n"
        
        def _msg_text_for_prompt(msg: dict) -> str:
            if "content" in msg: return str(msg["content"])
            parts = msg.get("parts")
            if isinstance(parts, list): return "".join(str(p) for p in parts)
            return str(parts) if parts is not None else ""

        formatted_history = "\n".join([("사용자: " if (m.get("role") == "user") else "진묘: ") + _msg_text_for_prompt(m) for m in chat_history])
        
        prompt = f"""당신은 국립공주박물관의 전문 AI 도슨트입니다.
당신의 임무는 반드시 아래 [이전 대화 내용]과 [참고 자료]에만 근거하여 사용자의 마지막 [질문]에 대해 답변하는 것입니다.
답변은 친절하고 이해하기 쉬운 설명체로 작성해주세요.
자료에 없는 내용은 절대로 지어내지 말고, "자료에 없는 내용이라 답변할 수 없습니다."라고 솔직하게 답변하세요.
[중요 규칙]
- 일반적인 유물 설명에는 절대로 '소장품번호'와 같은 내부 관리 번호를 포함하지 마세요.
- 단, 사용자가 '유물번호'나 '소장품번호'를 명시적으로 물어보는 경우에만, 참고 자료에 있는 '소장품번호' 값을 사용하여 답변해야 합니다.

---
[이전 대화 내용]
{formatted_history}
---
[참고 자료]
{context_for_llm.strip()}
---

[사용자 질문]
{query}

[답변]
"""
        try:
            response = self.llm_model.generate_content(prompt)
            return {"answer": response.text, "metadata": retrieved_docs}
        except Exception as e:
            return {"error": f"Gemini API 호출 중 오류 발생: {e}"}

chatbot_instance = RAGChatbot()

