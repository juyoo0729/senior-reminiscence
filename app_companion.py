# -*- coding: utf-8 -*-
"""app_companion.py — 시니어 AI 컴패니언 Streamlit UI (음성 입출력 추가).

실행:
    cd companion
    streamlit run app_companion.py

[음성 추가] 변경점은 주석 "# ★음성" 으로 표시.
- 입력: 마이크 녹음 → Whisper(STT) → 텍스트
- 출력: AI 답변 → OpenAI TTS → 음성 재생
- RAG 본체(분류/검색/생성)는 그대로.
[스토리북 추가] 변경점은 주석 "# ★스토리북" 으로 표시.
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
from storybook import render_storybook_mode  # ★스토리북: 스토리북 모드 UI

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

# ★UI개선: 카테고리 구분 없이 단일 리스트로 펼침 — 첫 화면 버튼에 사용
FLAT_EXAMPLES = [ex for exs in EXAMPLE_PROMPTS.values() for ex in exs]

# ── 페이지 설정 ────────────────────────────────────────────
st.set_page_config(page_title="마음 말동무", page_icon="🌿", layout="centered")

# ★UI개선: 모바일 친화 CSS — 글자·여백·터치 영역 전면 개선
st.markdown("""
<style>
/* ── 전체 컨테이너: 모바일 여백 정리 ── */
.block-container {
    padding-top: 1.2rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    padding-bottom: 5rem !important;
    max-width: 680px !important;
}

/* ── 채팅 말풍선 본문 24px ── */
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] {
    font-size: 24px !important;
    line-height: 2.0 !important;
}

/* ── 일반 본문 20px ── */
.stMarkdown p, .stMarkdown li {
    font-size: 20px !important;
    line-height: 1.95 !important;
}

/* ── 채팅 말풍선 여백·색 ── */
[data-testid="stChatMessage"] {
    padding: 20px 22px !important;
    border-radius: 18px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 2px 10px rgba(160, 120, 80, 0.12) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background-color: #FFF0DC !important;
    border-left: 4px solid #E8935A !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background-color: #F3F9ED !important;
    border-left: 4px solid #7AB68A !important;
}

/* ── 예시 버튼: 큰 터치 영역, 따뜻한 베이지 ── */
[data-testid="baseButton-secondary"] {
    font-size: 20px !important;
    font-weight: 600 !important;
    padding: 16px 20px !important;
    border-radius: 16px !important;
    background-color: #FEF0DC !important;
    color: #5C3317 !important;
    border: 2px solid #D4956A !important;
    line-height: 1.6 !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 64px !important;
    margin-bottom: 10px !important;
    text-align: left !important;
    width: 100% !important;
}
[data-testid="baseButton-secondary"]:hover {
    background-color: #EDCFAB !important;
    border-color: #B85C28 !important;
}

/* ── 참고 이야기 expander ── */
[data-testid="stExpander"] {
    border-radius: 12px !important;
    border: 1px solid #E8C9A8 !important;
    background-color: #FFFDF8 !important;
    margin-top: 8px !important;
}
[data-testid="stExpander"] summary span {
    font-size: 17px !important;
}

/* ★UI개선: st.chat_input 제거 → text_input + form_submit_button으로 교체 */

/* ── 텍스트 입력칸 ── */
.stTextInput input {
    font-size: 20px !important;
    padding: 14px 16px !important;
    border-radius: 14px !important;
    min-height: 56px !important;
    border: 2px solid #D4956A !important;
    background-color: #FFFBF4 !important;
    color: #3D2B1F !important;
    line-height: 1.8 !important;
}
.stTextInput input:focus {
    border-color: #C06C34 !important;
    box-shadow: 0 0 0 3px rgba(212, 149, 106, 0.28) !important;
}

/* ── 보내기 버튼 (form submit — primary) ── */
[data-testid="baseButton-primary"] {
    font-size: 20px !important;
    font-weight: 700 !important;
    padding: 14px !important;
    border-radius: 16px !important;
    background-color: #E8935A !important;
    color: white !important;
    border: none !important;
    min-height: 60px !important;
    margin-top: 6px !important;
    width: 100% !important;
}
[data-testid="baseButton-primary"]:hover {
    background-color: #C06C34 !important;
}

/* form 테두리 제거 */
[data-testid="stForm"] {
    border: none !important;
    padding: 0 !important;
}

/* ── 캡션·설명 텍스트 ── */
.stCaption p, [data-testid="stCaptionContainer"] p {
    font-size: 16px !important;
    color: #5C4033 !important;
}

/* ── 사이드바 ── */
[data-testid="stSidebar"] .stMarkdown p {
    font-size: 17px !important;
}
[data-testid="stSidebar"] .stCaption p {
    font-size: 15px !important;
    color: #5C4033 !important;
}

/* ── 섹션 제목 여백 ── */
h4, h5 {
    margin-top: 24px !important;
    margin-bottom: 10px !important;
}

/* ── 모드 선택 버튼: 크고 명확하게 ── */
.mode-btn button {
    font-size: 26px !important;
    font-weight: 700 !important;
    min-height: 90px !important;
    padding: 22px 16px !important;
    border-radius: 20px !important;
    white-space: normal !important;
    line-height: 1.5 !important;
    text-align: center !important;
    margin-bottom: 0 !important;
    width: 100% !important;
}
.mode-btn-active button {
    background-color: #4a7c59 !important;
    color: white !important;
    border: 3px solid #2d5c3e !important;
    box-shadow: 0 4px 14px rgba(74, 124, 89, 0.35) !important;
}
.mode-btn-inactive button {
    background-color: #F3F9ED !important;
    color: #2d5c3e !important;
    border: 3px solid #7AB68A !important;
}
.mode-btn-inactive button:hover {
    background-color: #D9EFE0 !important;
    border-color: #4a7c59 !important;
}
</style>
""", unsafe_allow_html=True)

# ★UI개선: 헤더 — 부제목 색 진하게(#5C4033), 모바일에서 줄바꿈 허용
st.markdown(
    "<h1 style='text-align:center; color:#4a7c59; font-size:2.0rem;"
    " margin-bottom:4px; line-height:1.3;'>🌿 마음 말동무</h1>"
    "<p style='text-align:center; color:#5C4033; font-size:1.05rem;"
    " margin-top:0; line-height:1.8;'>"
    "어르신을 위한 AI 말동무</p>",
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
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "chat"

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": GREETING}]

if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None

if "prompt_input" not in st.session_state:
    st.session_state.prompt_input = None

# ★스토리북: 마지막 분류 카테고리 — 스토리북 이미지 매칭에 사용
if "last_category" not in st.session_state:
    st.session_state.last_category = "감정긍정"

# ★음성: 이미 자동재생한 메시지 인덱스 (합창 방지)
if "played_audio_indices" not in st.session_state:
    st.session_state.played_audio_indices = set()

# ── 모드 선택 버튼 ─────────────────────────────────────────
mcol1, mcol2 = st.columns(2, gap="small")
with mcol1:
    chat_cls = "mode-btn mode-btn-active" if st.session_state.app_mode == "chat" else "mode-btn mode-btn-inactive"
    st.markdown(f'<div class="{chat_cls}">', unsafe_allow_html=True)
    if st.button("💬 말동무와 대화", key="mode_chat", use_container_width=True):
        st.session_state.app_mode = "chat"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
with mcol2:
    sb_cls = "mode-btn mode-btn-active" if st.session_state.app_mode == "storybook" else "mode-btn mode-btn-inactive"
    st.markdown(f'<div class="{sb_cls}">', unsafe_allow_html=True)
    if st.button("📖 내 이야기 스토리북", key="mode_storybook", use_container_width=True):
        st.session_state.app_mode = "storybook"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
st.divider()

# ── 모드별 콘텐츠 ──────────────────────────────────────────
if st.session_state.app_mode == "chat":
    # ── 대화 히스토리 표시 ─────────────────────────────────────
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            # ★음성: 새 메시지만 자동재생, 이미 재생된 건 수동(합창 방지)
            if msg.get("audio"):
                already_played = i in st.session_state.played_audio_indices
                st.audio(
                    msg["audio"],
                    format="audio/mp3",
                    autoplay=not already_played,
                )
                st.session_state.played_audio_indices.add(i)
            if msg.get("hits"):
                cat_label = CAT_LABEL.get(msg["category"], msg["category"])
                with st.expander(f"🔎 참고 이야기 ({cat_label})", expanded=False):
                    for h in msg["hits"]:
                        st.markdown(
                            f"- *(유사도 {h['score']:.3f})* `{h['label']}` {h['text'][:90]}"
                        )

    # ★UI개선: 예시 버튼 — 첫 화면엔 본문에 큰 버튼으로 전부 표시(카테고리 라벨 제거로 간소화)
    #           대화 시작 후엔 작은 접기 패널로 숨겨 화면 정리
    user_has_spoken = any(m["role"] == "user" for m in st.session_state.messages)

    if not user_has_spoken:
        st.markdown("#### 💬 이렇게 말씀해 보세요")
        for ex in FLAT_EXAMPLES:
            if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                st.session_state.prompt_input = ex
        st.divider()
    else:
        with st.expander("💬 다른 말 걸어보기", expanded=False):
            for ex in FLAT_EXAMPLES:
                if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                    st.session_state.prompt_input = ex

    # ── 입력 받기 (음성 + 텍스트) ──────────────────────────────
    st.markdown("##### 🎤 말로 하시거나, 아래에 적어주세요")
    # 마이크 버튼 가운데 정렬 — 기존 유지
    _left, _center, _right = st.columns([1, 2, 1])
    with _center:
        audio = mic_recorder(
            start_prompt="🎤 눌러서 말하기",
            stop_prompt="⏹️ 멈추기",
            just_once=True,
            format="wav",
            use_container_width=True,
            key="mic",
        )
        # ★UI개선: st.chat_input(화면 맨 아래 고정) → st.form(마이크 바로 아래 배치)
        # clear_on_submit=True: 제출 후 입력칸 자동 클리어, Enter키·버튼 모두 제출 처리
        with st.form(key="text_form", clear_on_submit=True):
            typed_raw = st.text_input(
                "직접 입력",
                placeholder="여기에 적어주세요...",
                label_visibility="collapsed",
            )
            send_clicked = st.form_submit_button(
                "📤 말 보내기",
                use_container_width=True,
                type="primary",
            )

    # ★UI개선: form 제출값을 typed_input으로 추출 (기존 typed_input 변수명 유지)
    typed_input = typed_raw.strip() if send_clicked and typed_raw.strip() else None

    # ★음성: 입력 소스 결정 — 녹음이 있으면 STT로 텍스트화, 없으면 타이핑 사용
    user_input = None
    if st.session_state.prompt_input:
        user_input = st.session_state.prompt_input
        st.session_state.prompt_input = None
    elif audio and audio.get("id") != st.session_state.last_audio_id:
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
        st.session_state.last_category = result["category"]  # ★스토리북: 카테고리 기억
        st.rerun()

elif st.session_state.app_mode == "storybook":
    render_storybook_mode()  # ★스토리북: "준비 중" placeholder → 실제 기능

# ★UI개선: 사이드바 간소화 — 예시 목록 제거(본문 버튼으로 이동), 초기화·출처만 유지
with st.sidebar:
    turn_count = sum(1 for m in st.session_state.messages if m["role"] == "user")
    if turn_count:
        remembered = min(turn_count, config.MAX_HISTORY_TURNS)
        st.caption(f"대화 {turn_count}회 · {remembered}턴 기억 중")
        st.divider()

    if st.button("🔄 대화 초기화", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": GREETING}]
        st.session_state.last_audio_id = None
        st.session_state.prompt_input = None
        st.session_state.storybook = None
        st.session_state.played_audio_indices = set()  # ← 추가
        st.rerun()

    st.divider()
    st.caption("AI허브 '고령자 근현대 경험 기반\n스토리 구술 데이터' 기반")
