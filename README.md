무령왕릉 도슨트 '진묘'

RAG(검색 증강 생성)와 Google Gemini API를 기반으로 제작된, 백제 무령왕릉 전문 AI 도슨트 챗봇 '진묘'입니다.

이 프로젝트는 사용자가 무령왕릉 관련 유물 및 역사에 대해 질문하면, 제공된 전문 데이터(국립공주박물관 보고서 등)에 기반하여 신뢰도 높은 답변을 제공하는 대화형 AI 웹 애플리케이션입니다.

✨ 주요 기능
대화형 RAG: 사용자의 질문 의도를 파악하고, 자체 구축한 벡터 데이터베이스에서 가장 관련성 높은 정보를 찾아내어 답변을 생성합니다.

지능형 라우팅: '시맨틱 라우터'를 도입하여, 단순 키워드를 넘어 질문의 숨은 의도(상세 정보, 비교, 역사 배경 등)를 파악하고 최적의 검색 전략을 수행합니다.

대화 맥락 이해: 이전 대화 기록을 기억하고, "그건 뭐야?"와 같은 대명사나 후속 질문의 맥락을 이해하여 자연스러운 대화를 이어갑니다.

이미지 통합 답변: 유물에 대한 설명과 함께, 관련 이미지를 채팅창에 직접 표시하여 사용자 경험을 향상시킵니다.

인터랙티브 사이드바: 대화 중에 언급된 유물들이 사이드바에 자동으로 기록되며, 사용자는 언제든지 목록의 유물을 클릭하여 다시 자세한 정보를 확인할 수 있습니다.

추천 질문: 처음 방문한 사용자를 위해, 흥미로운 질문 예시를 버튼으로 제공하여 상호작용을 유도합니다.

🛠️ 기술 스택
웹 프레임워크: Streamlit

핵심 AI 모델:

생성(LLM): Google Gemini 1.5 Pro

임베딩: upskyy/bge-m3-korean (장문 처리 특화)

검색 엔진: FAISS (Facebook AI Similarity Search)

주요 라이브러리: pandas, PyMuPDF, sentence-transformers, pillow

데이터 소스: 국립공주박물관 제공 데이터 및 보고서

📂 프로젝트 구조
Muryeong-RAG-Bot/
├── data/
│   ├── converted_검증_통합.csv
│   ├── preprocessed_artifacts_final_with_images.csv
│   ├── preprocessed_history_chunks_sectioned.csv
│   ├── pdf_data/
│   └── extracted_images/
├── vector_store/
│   ├── artifacts.index
│   ├── artifacts_df.pkl
│   ├── history.index
│   └── history_df.pkl
├── .env                 # (API 키 - 로컬 전용)
├── chatbot.py           # 챗봇 핵심 로직
├── config.py            # 설정 관리
├── data_preprocessor.py # CSV 데이터 정제
├── pdf_processor.py     # PDF 데이터 정제
├── vector_store_builder.py # 벡터 DB 구축
├── streamlit_app.py     # Streamlit 웹 앱
├── requirements.txt     # 라이브러리 목록
└── README.md            # 프로젝트 설명

🚀 시작하기
1. 사전 준비
Python 3.9 이상

Google AI Studio에서 발급받은 Gemini API 키

2. 설치
저장소 복제:

git clone https://github.com/YOUR_USERNAME/Muryeong-RAG-Bot.git
cd Muryeong-RAG-Bot

가상 환경 생성 및 활성화:

# 가상 환경 생성
python -m venv .venv
# 활성화 (Windows)
.\.venv\Scripts\Activate
# 활성화 (macOS/Linux)
source .venv/bin/activate

필요한 라이브러리 설치:

pip install -r requirements.txt

API 키 설정:

프로젝트 루트 디렉토리에 .env 파일을 생성합니다.

파일을 열고 아래 내용을 작성한 후, 자신의 Gemini API 키를 붙여넣습니다.

GEMINI_API_KEY="YOUR_GOOGLE_AI_API_KEY"

3. 데이터 파이프라인 실행
챗봇을 실행하기 전에, 원본 데이터를 가공하여 벡터 데이터베이스를 구축해야 합니다. 아래 스크립트를 순서대로 실행해주세요.

CSV 데이터 정제:

python data_preprocessor.py

PDF 데이터 정제:

python pdf_processor.py

벡터 DB 구축 (시간 소요):

python vector_store_builder.py

4. 챗봇 실행
모든 준비가 완료되었습니다. 아래 명령어로 Streamlit 웹 앱을 실행합니다.

streamlit run streamlit_app.py

웹 브라우저에서 http://localhost:8501 주소로 접속하여 '진묘'와 대화를 시작할 수 있습니다.

☁️ 배포
이 애플리케이션은 Streamlit Community Cloud를 통해 쉽게 배포할 수 있습니다.

프로젝트를 GitHub 공개(Public) 저장소에 업로드합니다. (단, .env 파일은 제외)

Streamlit Community Cloud에 접속하여 New app을 선택하고, 해당 저장소를 연결합니다.

**Advanced settings...**의 Secrets 섹션에 API 키를 추가합니다:

GEMINI_API_KEY="YOUR_GOOGLE_AI_API_KEY"

Deploy! 버튼을 클릭합니다.
