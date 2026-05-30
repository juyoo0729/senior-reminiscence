# -*- coding: utf-8 -*-
"""app_companion.py — 시니어 AI 컴패니언 Streamlit UI (음성 입출력 추가).

실행:
    cd companion
    streamlit run app_companion.py

[음성 추가] 변경점은 주석 "# ★음성" 으로 표시.
- 입력: 마이크 녹음 → Whisper(STT) → 텍스트
- 출력: AI 답변 → OpenAI TTS → 음성 재생
- RAG 본체(분류/검색/생성)는 그대로.
"""
import os
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from src.pipeline import build_retriever
from src import generator, config
from src import voice  # ★음성: STT/TTS 함수
from streamlit_mic_recorder import mic_recorder  # ★음성: 마이크 녹음 컴포넌트

# ── 카테고리 → 이모지/레이블 매핑 ──────────────────────────
CAT_LABEL = {
    "감정긍정": "😊 기쁜 이야기",
    "감정부정": "💙 힘드셨군요",
    "사물":   "🏠 물건 이야기",
    "장소":   "🗺️ 장소 이야기",
    "관계":   "👥 관계 이야기",
}

GREETING = (
    "안녕하세요! 저는 어르신 말동무입니다. 😊\n\n"
    "오늘 하루 어떠셨나요? 기쁜 일이든 힘든 일이든 편하게 말씀해 주세요."
)

EXAMPLE_PROMPTS = {
    "외로움·그리움": [
        "밤에 혼자 있으면 무섭고 쓸쓸해",
        "자식들이 바빠서 통 연락이 없네",
        "젊었을 때 친구들이 그립구먼",
    ],
    "기쁜 이야기": [
        "어제 손주가 왔다 갔는데 너무 예뻐",
        "오늘 공원에서 산책했더니 기분이 좋아",
    ],
    "옛날 이야기": [
        "예전에 장사하던 게 그립구먼",
        "젊었을 때 고향 밥이 제일 맛있었어",
    ],
}

# ── 페이지 설정 ────────────────────────────────────────────
st.set_page_config(page_title="마음 말동무", page_icon="🌿", layout="centered")

# ── 어르신 친화 CSS ────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] {
    font-size: 20px !important;
    line-height: 1.9 !important;
}
.stMarkdown p, .stMarkdown li {
    font-size: 18px !important;
    line-height: 1.8;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li {
    font-size: 17px !important;
}
[data-testid="stSidebar"] .stCaption p {
    font-size: 15px !important;
}
[data-testid="stExpander"] summary span {
    font-size: 16px !important;
}
[data-testid="stChatMessage"] {
    padding: 18px 22px !important;
    border-radius: 18px !important;
    margin-bottom: 14px !important;
    box-shadow: 0 2px 10px rgba(180, 140, 100, 0.12) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background-color: #FFF0DC !important;
    border-left: 4px solid #E8935A !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background-color: #F3F9ED !important;
    border-left: 4px solid #7AB68A !important;
}
[data-testid="stExpander"] {
    border-radius: 12px !important;
    border: 1px solid #E8C9A8 !important;
    background-color: #FFFDF8 !important;
    margin-top: 8px !important;
}
.stChatInput textarea {
    font-size: 19px !important;
    padding: 14px 18px !important;
    border-radius: 14px !important;
    min-height: 58px !important;
    border: 2px solid #D4956A !important;
    background-color: #FFFBF4 !important;
    color: #3D2B1F !important;
}
.stChatInput textarea:focus {
    border-color: #C06C34 !important;
    box-shadow: 0 0 0 3px rgba(212, 149, 106, 0.30) !important;
}
.stChatInput button {
    background-color: #E8935A !important;
    border-radius: 12px !important;
    padding: 10px 14px !important;
}
.stChatInput button:hover {
    background-color: #C06C34 !important;
}
[data-testid="baseButton-secondary"] {
    font-size: 17px !important;
    padding: 12px !important;
    border-radius: 12px !important;
    background-color: #F5E6D3 !important;
    color: #7B4E2D !important;
    border: 1.5px solid #D4956A !important;
    font-weight: 600 !important;
}
[data-testid="baseButton-secondary"]:hover {
    background-color: #E8C9A8 !important;
}
</style>
""", unsafe_allow_html=True)

# ── 헤더 ──────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center; color:#4a7c59; font-size:2.1rem; margin-bottom:4px;'>"
    "🌿 마음 말동무</h1>"
    "<p style='text-align:center; color:#8A7060; font-size:1.05rem; margin-top:0;'>"
    "홀로 지내는 어르신을 위한 AI 말동무 · 실제 어르신들의 이야기를 근거로 공감합니다"
    "</p>",
    unsafe_allow_html=True,
)
st.divider()

if not os.environ.get("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY가 설정되지 않았습니다. .env 또는 환경변수를 확인하세요.")
    st.stop()


@st.cache_resource(show_spinner="모델을 불러오는 중입니다...")
def load_pipeline():
    return build_retriever()


retrieve_fn = load_pipeline()

# ── 세션 초기화 ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": GREETING}]

# ★음성: 같은 녹음을 두 번 처리하지 않도록 마지막 녹음 ID 기억
if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None

# ── 대화 히스토리 표시 ─────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # ★음성: 저장해둔 답변 음성이 있으면 재생 위젯 표시
        if msg.get("audio"):
            st.audio(msg["audio"], format="audio/mp3")
        if msg.get("hits"):
            cat_label = CAT_LABEL.get(msg["category"], msg["category"])
            with st.expander(f"🔎 참고 이야기 ({cat_label})", expanded=False):
                for h in msg["hits"]:
                    st.markdown(
                        f"- *(유사도 {h['score']:.3f})* `{h['label']}` {h['text'][:90]}"
                    )

# ── 입력 받기 (음성 + 텍스트) ──────────────────────────────
# ★음성: 마이크 녹음 버튼. 녹음하면 dict(bytes, id, ...) 반환.
st.markdown("##### 🎤 말로 하시거나, 아래에 적어주세요")
audio = mic_recorder(
    start_prompt="🎤 눌러서 말하기",
    stop_prompt="⏹️ 멈추기",
    just_once=True,
    format="wav",
    key="mic",
)

# 텍스트 입력 (기존)
typed_input = st.chat_input("편하게 말씀해 주세요...")

# ★음성: 입력 소스 결정 — 녹음이 있으면 STT로 텍스트화, 없으면 타이핑 사용
user_input = None
if audio and audio.get("id") != st.session_state.last_audio_id:
    st.session_state.last_audio_id = audio["id"]
    with st.spinner("말씀을 듣고 있어요..."):
        try:
            user_input = voice.speech_to_text(audio["bytes"])
        except Exception as e:
            st.error(f"음성 인식에 실패했어요: {e}")
            user_input = None
elif typed_input:
    user_input = typed_input

# ── 입력 처리 (음성·텍스트 공통) ───────────────────────────
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # 분류 + 검색
    with st.spinner("잠시만요..."):
        result = retrieve_fn(user_input)

    # 대화 히스토리 구성
    _max = config.MAX_HISTORY_TURNS * 2
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
        if m["content"] != GREETING
    ][-_max:]

    # 스트리밍 생성
    with st.chat_message("assistant"):
        response_stream = generator.generate_stream(
            user_input, result["hits"], result["category"], history
        )
        answer = st.write_stream(response_stream)

        # ★음성: 답변을 음성으로 변환해서 재생
        audio_bytes = None
        with st.spinner("목소리로 만들고 있어요..."):
            try:
                audio_bytes = voice.text_to_speech(answer)
            except Exception as e:
                st.warning(f"음성 변환은 건너뛰었어요: {e}")
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3", autoplay=True)

        cat_label = CAT_LABEL.get(result["category"], result["category"])
        with st.expander(f"🔎 참고 이야기 ({cat_label})", expanded=False):
            for h in result["hits"]:
                st.markdown(
                    f"- *(유사도 {h['score']:.3f})* `{h['label']}` {h['text'][:90]}"
                )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "category": result["category"],
        "hits": result["hits"],
        "audio": audio_bytes,  # ★음성: 다시 그릴 때 재생되도록 저장
    })

# ── 사이드바 ───────────────────────────────────────────────
with st.sidebar:
    st.subheader("💬 이렇게 말 걸어보세요")
    for group, examples in EXAMPLE_PROMPTS.items():
        st.markdown(f"**{group}**")
        for ex in examples:
            st.markdown(f"- {ex}")
        st.markdown("")

    turn_count = sum(1 for m in st.session_state.messages if m["role"] == "user")
    if turn_count:
        remembered = min(turn_count, config.MAX_HISTORY_TURNS)
        st.caption(f"대화 {turn_count}회 · {remembered}턴 기억 중")

    st.divider()
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": GREETING}]
        st.session_state.last_audio_id = None  # ★음성
        st.rerun()

    st.divider()
    st.caption("AI허브 '고령자 근현대 경험 기반 스토리 구술 데이터' 기반")
