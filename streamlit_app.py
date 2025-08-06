# streamlit_app.py
import streamlit as st
from chatbot import chatbot_instance 
from PIL import Image 
import os

# --- 페이지 기본 설정 및 아바타 이미지 로드 ---
try:
    jinmyo_avatar = Image.open("favicon.png")
except FileNotFoundError:
    jinmyo_avatar = "👑" 

st.set_page_config(
    page_title="무령왕릉 도슨트 '진묘'",
    page_icon=jinmyo_avatar,
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS (다크 모드) ---
st.markdown("""
<style>
    /* ... (이전과 동일한 CSS 코드) ... */
    .stApp { background-color: #0E1117; }
    [data-testid="main-container"] { background-color: #161A21; border-radius: 10px; padding: 2rem; box-shadow: 0 8px 32px rgba(0,0,0,0.3); }
    .sidebar-container { background-color: #0E1117; border: 1px solid #262730; border-radius: 10px; padding: 1.5rem 1rem; height: 100%; }
    h1, h2, h3, h4, h5, h6 { color: #FAFAFA; } .stCaption { color: #A0A0A0; }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageContent-user"]) { background-color: #4F46E5; border-radius: 15px; }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageContent-assistant"]) { background-color: #374151; border-radius: 15px; }
    .stButton>button { border-radius: 8px; border: 1px solid #4B5563; background-color: #374151; color: #F3F4F6; transition: all 0.2s; font-weight: 500; }
    .stButton>button:hover { background-color: #4B5563; border-color: #6B7280; color: #FFFFFF; }
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
    
    gemini_history = []
    for msg in st.session_state.chat_history[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [msg["content"]]})

    result = chatbot_instance.ask(prompt, gemini_history)
    
    assistant_response = {"role": "assistant"}
    if "error" in result:
        assistant_response["content"] = f"죄송해요, 오류가 발생했어요: {result['error']}"
    else:
        response_text = result["answer"]
        if result.get("metadata"):
            meta = result["metadata"][0]
            
            # (⭐ 핵심 수정) 로컬 파일 경로로 이미지 경로 생성 및 확인
            if meta.get("image_url"):
                # '/static/images/...' 에서 파일 이름만 추출
                file_name = meta["image_url"].split('/')[-1]
                # 내 컴퓨터의 실제 파일 경로 구성
                local_image_path = os.path.join("data", "extracted_images", file_name)
                
                # 해당 경로에 파일이 실제로 존재하는지 확인
                if os.path.exists(local_image_path):
                    assistant_response["image"] = local_image_path
                else:
                    print(f"이미지 파일을 찾을 수 없습니다: {local_image_path}")


            links = []
            if meta.get("MUCH_URL"): links.append(f"[자세히 보기]({meta['MUCH_URL']})")
            if links: response_text += "\n\n---\n" + " | ".join(links)
        
        assistant_response["content"] = response_text
            
    st.session_state.chat_history.append(assistant_response)

    if result.get("metadata"):
        for meta in result["metadata"]:
            if meta.get("id") and not meta.get("source_file"):
                st.session_state.mentioned_artifacts[meta["id"]] = meta["명칭"]

# --- UI 렌더링 ---

with st.container():
    st.markdown('<div data-testid="main-container">', unsafe_allow_html=True)
    main_col, sidebar_col = st.columns([2.5, 1])

    with main_col:
        col1, col2 = st.columns([0.15, 0.85])
        with col1:
            if isinstance(jinmyo_avatar, Image.Image): st.image(jinmyo_avatar, width=80)
        with col2:
            st.title("무령왕릉 도슨트 '진묘'"); st.caption("안녕하세요! 저는 백제 무령왕릉을 지키는 도슨트 '진묘'입니다.")
        st.markdown("---")

        chat_container = st.container(height=500, border=False)
        with chat_container:
            for message in st.session_state.chat_history:
                avatar_to_use = "🧑‍💻" if message["role"] == "user" else jinmyo_avatar
                with st.chat_message(message["role"], avatar=avatar_to_use):
                    if message["role"] == "assistant" and "image" in message:
                        st.image(message["image"]) # 이제 이 경로는 로컬 파일 경로입니다.
                    st.markdown(message["content"])
        
        if not st.session_state.chat_history:
            st.markdown("##### ✨ 이런 질문은 어떠세요?")
            suggested_questions = ["무령왕릉은 언제, 어떻게 발견되었나요?", "진묘수에 대해 자세히 알려주세요.", "왕의 귀걸이는 어떻게 생겼어?"]
            cols = st.columns(len(suggested_questions))
            for i, q in enumerate(suggested_questions):
                with cols[i]:
                    if st.button(q, use_container_width=True, key=f"suggest_{i}"):
                        handle_query(q); st.rerun()
        
        if prompt := st.chat_input("진묘에게 무엇이든 물어보세요..."):
            handle_query(prompt); st.rerun()

    with sidebar_col:
        st.markdown('<div class="sidebar-container">', unsafe_allow_html=True)
        st.markdown("#### 📜 언급된 유물 목록")
        st.markdown("---")
        if not st.session_state.mentioned_artifacts:
            st.info("아직 대화에 언급된 유물이 없습니다.")
        else:
            for artifact_id, artifact_name in st.session_state.mentioned_artifacts.items():
                if st.button(artifact_name, key=f"artifact_{artifact_id}", use_container_width=True):
                    handle_query(f"{artifact_name}에 대해 자세히 알려줘."); st.rerun()
        if st.button("새 대화 시작", use_container_width=True, key="new_chat_sidebar"):
            st.session_state.chat_history = []; st.session_state.mentioned_artifacts = {}; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)