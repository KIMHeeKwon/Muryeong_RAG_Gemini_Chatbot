# app.py
from flask import Flask, request, jsonify, render_template, session
from dotenv import load_dotenv
import os
import google.generativeai as genai
from chatbot import chatbot_instance

load_dotenv()
app = Flask(__name__)
# (⭐ 핵심 추가) 세션 기능을 사용하기 위한 시크릿 키 설정
app.secret_key = os.urandom(24)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
genai.configure(api_key=api_key)

@app.route('/')
def home():
    session.clear() # 메인 페이지 접속 시 대화 기록 초기화
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_api():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "질문(query)이 없습니다."}), 400

    query = data['query']
    # (⭐ 핵심 수정) 세션에서 대화 기록 가져오기
    chat_history = session.get('chat_history', [])

    # 수정된 ask 함수에 대화 기록 전달
    result = chatbot_instance.ask(query, chat_history)
    
    # (⭐ 핵심 수정) 대화 기록 업데이트 및 세션에 저장
    if 'error' not in result:
        chat_history.append({"role": "user", "parts": [query]})
        chat_history.append({"role": "model", "parts": [result.get('answer', '')]})
        session['chat_history'] = chat_history

    return jsonify(result)

# (⭐ 핵심 추가) 대화 기록 초기화 API
@app.route('/clear', methods=['POST'])
def clear_history():
    session.clear()
    return jsonify({"status": "cleared"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)