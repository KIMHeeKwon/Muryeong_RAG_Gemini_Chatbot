# streamlit_app.py
import streamlit as st
from chatbot import chatbot_instance 
from PIL import Image 
import os

# --- 페이지 기본 설정 ---
try:
    jinmyo_avatar = Image.open("favicon.png")
except FileNotFoundError:
    jinmyo_avatar = "👑" 

st.set_page_config(
    page_title="무령왕릉 도슨트 '진묘'",
    page_icon=jinmyo_avatar,
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Custom CSS (밝은 스타일) ---
st.markdown("""
<style>
    /* ... (CSS는 이전과 동일) ... */
    .stApp { background-color: #FFFFFF; }
    [data-testid="stSidebar"] { background-color: #F0F2F5; border-right: 1px solid #E0E0E0; }
    h1, h2, h3, h4, h5, h6 { color: #111827; } .stCaption { color: #6B7280; }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageContent-user"]) { background-color: #DBEAFE; }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageContent-assistant"]) { background-color: #F1F1F1; }
    .stButton>button { border-radius: 8px; border: 1px solid #D1D5DB; background-color: #FFFFFF; color: #374151; }
    .stButton>button:hover { background-color: #F9FAFB; border-color: #6B7280; }
    [data-testid="stChatInput"] { background-color: #FFFFFF; }
</style>
""", unsafe_allow_html=True)


# --- 세션 상태 초기화 ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "mentioned_artifacts" not in st.session_state:
    st.session_state.mentioned_artifacts = {}

# --- 공통 함수 정의 ---
def handle_query(prompt):
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Gemini API에 전달할 대화 기록을 정확한 형식으로 변환
    gemini_history = []
    for msg in st.session_state.chat_history[:-1]:
        role = "user" if msg.get("role") == "user" else "model"
        content = msg.get("content", "")
        gemini_history.append({"role": role, "parts": [content]})
    
    result = chatbot_instance.ask(prompt, gemini_history)

    # (⭐ 핵심 수정 1) 답변과 모든 관련 메타데이터를 하나의 딕셔너리로 묶어 저장
    assistant_response = {
        "role": "assistant",
        "content": "",
        "metadata": [] # 메타데이터를 담을 빈 리스트
    }
    if "error" in result:
        assistant_response["content"] = f"죄송해요, 오류가 발생했어요:\n{result['error']}"
    else:
        assistant_response["content"] = result.get("answer", "")
        # 검색된 모든 메타데이터를 그대로 저장하여 텍스트와 정보가 분리되지 않도록 함
        assistant_response["metadata"] = result.get("metadata", [])
            
    st.session_state.chat_history.append(assistant_response)

    # 유물 목록 업데이트
    if result.get("metadata"):
        for meta in result["metadata"]:
            if meta.get("id") and not meta.get("source_file"):
                st.session_state.mentioned_artifacts[meta["id"]] = meta["명칭"]

# --- UI 렌더링 ---

# 사이드바 UI 구성
with st.sidebar:
    col1, col2 = st.columns([0.3, 0.7])
    with col1:
        if isinstance(jinmyo_avatar, Image.Image):
            st.image(jinmyo_avatar)
    with col2:
        st.header("무령왕릉 도슨트 '진묘'")
    
    if st.button("새 대화 시작", use_container_width=True, key="new_chat_sidebar"):
        st.session_state.chat_history = []
        st.session_state.mentioned_artifacts = {}
        st.rerun()
        
    st.markdown("---")
    st.markdown("#### 📜 언급된 유물 목록")
    if not st.session_state.mentioned_artifacts:
        st.info("아직 대화에 언급된 유물이 없습니다.")
    else:
        for artifact_id, artifact_name in st.session_state.mentioned_artifacts.items():
            if st.button(artifact_name, key=f"artifact_{artifact_id}", use_container_width=True):
                handle_query(f"{artifact_name}에 대해 자세히 알려줘."); st.rerun()

# 메인 채팅 화면 구성
if not st.session_state.chat_history:
    st.title("무령왕릉 도슨트 '진묘'")
    st.caption("안녕하세요! 저는 백제 무령왕릉에 대해서 알려주는 도슨트 '진묘'입니다.")
    st.markdown("---")
    
    st.markdown("##### ✨ 이런 질문은 어떠세요?")
    suggested_questions = ["무령왕릉은 언제, 어떻게 발견되었나요?", "진묘수에 대해 자세히 알려주세요.", "왕의 귀걸이는 어떻게 생겼어?"]
    
    for q in suggested_questions:
        if st.button(q, use_container_width=True, key=q):
            handle_query(q)
            st.rerun()

# (⭐ 핵심 수정 2) 저장된 메시지 '묶음'에서 직접 정보를 꺼내어 표시
for message in st.session_state.chat_history:
    avatar_to_use = "🧑‍💻" if message["role"] == "user" else jinmyo_avatar
    with st.chat_message(message["role"], avatar=avatar_to_use):
        # 텍스트 내용 표시
        st.markdown(message.get("content", ""))
        
        # 챗봇 메시지이고, 유물 메타데이터가 있다면 이미지와 링크 표시
        if message["role"] == "assistant" and message.get("metadata"):
            # 이 메시지에 묶여있는 메타데이터 사용
            # 유물 정보일 경우에만 (source_file 키가 없는 경우)
            if message["metadata"] and not message["metadata"][0].get("source_file"):
                meta = message["metadata"][0]
                # 이미지 표시
                if meta.get("image_url"):
                    file_name = meta["image_url"].split('/')[-1]
                    local_image_path = os.path.join("data", "extracted_images", file_name)
                    if os.path.exists(local_image_path):
                        st.image(local_image_path)
                # 링크 표시
                if meta.get("MUCH_URL"):
                    st.markdown(f"---\n[자세히 보기]({meta['MUCH_URL']})")

# 사용자 입력 처리
if prompt := st.chat_input("진묘에게 무엇이든 물어보세요..."):
    handle_query(prompt)
    st.rerun()

