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
    layout="centered", # (⭐ 핵심 수정) 레이아웃을 중앙 정렬로 변경하여 UI 오류 해결
    initial_sidebar_state="expanded",
)

# --- (⭐ 핵심 수정) Custom CSS (밝은 스타일) ---
st.markdown("""
<style>
    /* 전체 배경 */
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* 사이드바 스타일 */
    [data-testid="stSidebar"] {
        background-color: #F0F2F5;
        border-right: 1px solid #E0E0E0;
    }
    
    /* 제목 및 캡션 색상 */
    h1, h2, h3, h4, h5, h6 { color: #111827; }
    .stCaption { color: #6B7280; }

    /* 사용자 채팅 버블 */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageContent-user"]) {
        background-color: #DBEAFE; /* 밝은 파란색 */
    }

    /* 챗봇 채팅 버블 */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageContent-assistant"]) {
        background-color: #F1F1F1; /* 밝은 회색 */
    }
    
    /* 추천/사이드바 버튼 스타일 */
    .stButton>button {
        border-radius: 8px;
        border: 1px solid #D1D5DB;
        background-color: #FFFFFF;
        color: #374151;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #F9FAFB;
        border-color: #6B7280;
    }

    /* 채팅 입력창 스타일 */
    [data-testid="stChatInput"] {
        background-color: #FFFFFF;
    }
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

    # ✅ 히스토리를 재가공하지 않고, 직전까지의 대화만 그대로 전달
    result = chatbot_instance.ask(prompt, st.session_state.chat_history[:-1])

    assistant_response = {"role": "assistant"}
    if "error" in result:
        # ✅ 여러 줄 문자열이 줄바꿈으로 끊어지지 않도록 한 줄로
        assistant_response["content"] = f"죄송해요, 오류가 발생했어요:\n{result['error']}"
    else:
        response_text = result["answer"]
        if result.get("metadata"):
            meta = result["metadata"][0]
            if meta.get("image_url"):
                file_name = meta["image_url"].split('/')[-1]
                local_image_path = os.path.join("data", "extracted_images", file_name)
                if os.path.exists(local_image_path):
                    assistant_response["image"] = local_image_path
            links = []
            if meta.get("MUCH_URL"):
                links.append(f"[자세히 보기]({meta['MUCH_URL']})")
            if links:
                response_text += "\n\n---\n" + " | ".join(links)
        assistant_response["content"] = response_text

    st.session_state.chat_history.append(assistant_response)

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
        st.title("MUCH")
    
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

# (⭐ 핵심 수정) 메인 채팅 화면 구성 (st.columns 제거)
# 대화가 없을 때만 환영 메시지 및 추천 질문 표시
if not st.session_state.chat_history:
    st.title("무령왕릉 도슨트 '진묘'")
    st.caption("안녕하세요! 저는 백제 무령왕릉을 지키는 도슨트 '진묘'입니다.")
    st.markdown("---")
    
    st.markdown("##### ✨ 이런 질문은 어떠세요?")
    suggested_questions = ["무령왕릉은 언제, 어떻게 발견되었나요?", "진묘수에 대해 자세히 알려주세요.", "왕의 귀걸이는 어떻게 생겼어?"]
    
    for q in suggested_questions:
        if st.button(q, use_container_width=True, key=q):
            handle_query(q)
            st.rerun()

# 대화 기록 표시
for message in st.session_state.chat_history:
    avatar_to_use = "🧑‍💻" if message["role"] == "user" else jinmyo_avatar
    with st.chat_message(message["role"], avatar=avatar_to_use):
        if message["role"] == "assistant" and "image" in message:
            st.image(message["image"])
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("진묘에게 무엇이든 물어보세요..."):
    handle_query(prompt)
    st.rerun()
