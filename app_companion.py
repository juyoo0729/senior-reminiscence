# -*- coding: utf-8 -*-
"""app_companion.py — 시니어 AI 컴패니언 Streamlit UI (새 구조).

실행:
    cd companion
    streamlit run app_companion.py
"""
import os
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from src.pipeline import build_companion


@st.cache_resource
def load_app():
    return build_companion()


st.set_page_config(page_title="마음 말동무", page_icon="🌿", layout="centered")
st.title("🌿 마음 말동무")
st.caption("홀로 지내는 어르신을 위한 AI 말동무 · 실제 어르신들의 이야기를 근거로 공감합니다")

if not os.environ.get("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY가 설정되지 않았습니다. .env 또는 환경변수를 확인하세요.")
    st.stop()

app = load_app()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("hits"):
            with st.expander(f"🔎 근거 보기 (분류: {msg['category']})"):
                for h in msg["hits"]:
                    st.markdown(f"- *(유사도 {h['score']:.3f})* `{h['label']}` {h['text'][:80]}")

user_input = st.chat_input("편하게 말씀해 주세요...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    with st.chat_message("assistant"):
        with st.spinner("마음을 헤아리는 중..."):
            result = app.invoke({"user_input": user_input})
        st.write(result["answer"])
        with st.expander(f"🔎 근거 보기 (분류: {result['category']})"):
            for h in result["hits"]:
                st.markdown(f"- *(유사도 {h['score']:.3f})* `{h['label']}` {h['text'][:80]}")
    st.session_state.messages.append({
        "role": "assistant", "content": result["answer"],
        "category": result["category"], "hits": result["hits"],
    })

with st.sidebar:
    st.subheader("💬 이렇게 말 걸어보세요")
    st.markdown("- 밤에 혼자 있으면 무섭고 쓸쓸해\n- 자식들이 바빠서 통 연락이 없네\n"
                "- 어제 손주가 왔다 갔는데 예뻐\n- 옛날에 장사하던 게 그립구먼")
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("AI허브 '고령자 근현대 경험 기반 스토리 구술 데이터' 기반")
