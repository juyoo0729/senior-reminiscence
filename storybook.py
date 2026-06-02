# -*- coding: utf-8 -*-
"""storybook.py — 마음 말동무 스토리북 모듈.

대화 기록 → GPT가 1인칭 회상 이야기 생성 → 코드가 페이지 분할 →
카테고리별 저장 이미지 매칭 → 페이지 넘기기 UI.

설계 의도:
- GPT는 '이야기 생성'에만 집중 (창의적 부분).
- 페이지 분할/이미지 매칭은 코드가 결정론적으로 처리 (안정성 보장).
- 이미지는 실시간 생성 X, 카테고리별 저장본에서 선택 (비용 0·속도·안정성).
"""
import os
import re
import random

import streamlit as st
from openai import OpenAI

client = OpenAI()  # OPENAI_API_KEY 환경변수 사용

# ── 카테고리 → 이미지 폴더 매핑 ────────────────────────────
# images/positive/, images/negative/ ... 각 폴더에 ~10장씩
CATEGORY_DIRS = {
    "감정긍정": "positive",
    "감정부정": "negative",
    "사물":   "object",
    "장소":   "place",
    "관계":   "relation",
}
IMAGE_ROOT = os.path.join(os.path.dirname(__file__), "images")


def generate_story(conversation: list[dict]) -> str:
    """대화 기록 → 1인칭 회상 이야기 전문 생성."""
    user_turns = "\n".join(
        f"- {m['content']}" for m in conversation if m["role"] == "user"
    )

    prompt = f"""다음은 어르신과 나눈 대화에서 어르신이 들려주신 이야기들입니다.

{user_turns}

이 내용을 바탕으로, 어르신의 시점(1인칭)에서 따뜻하고 잔잔한 회상 이야기를 써 주세요.
- 어르신이 실제로 말씀하신 내용만 사용하고, 없는 사실을 지어내지 마세요.
- 문단을 3~5개로 나눠 주세요. 각 문단은 하나의 장면/기억 단위가 되도록 합니다.
- 문단 사이는 빈 줄로 구분해 주세요.
- 제목은 달지 말고 본문만 써 주세요."""

    resp = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


def split_pages(story: str) -> list[str]:
    """이야기 전문 → 페이지(문단) 리스트. 빈 줄 기준 분할(결정론적)."""
    return [p.strip() for p in re.split(r"\n\s*\n", story) if p.strip()]


def pick_image(category: str, used: set) -> str | None:
    """카테고리 폴더에서 아직 안 쓴 이미지 1장 선택."""
    folder = CATEGORY_DIRS.get(category)
    if not folder:
        return None
    dir_path = os.path.join(IMAGE_ROOT, folder)
    if not os.path.isdir(dir_path):
        return None
    candidates = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        and os.path.join(dir_path, f) not in used
    ]
    if not candidates:
        return None
    chosen = random.choice(candidates)
    used.add(chosen)
    return chosen


def build_storybook(conversation: list[dict], category: str) -> list[dict]:
    """대화 → 페이지 리스트 [{text, image}]."""
    story = generate_story(conversation)
    pages = split_pages(story)

    used = set()
    book = []
    for text in pages:
        book.append({"text": text, "image": pick_image(category, used)})
    return book


# ── 스토리북 모드 UI ───────────────────────────────────────
def render_storybook_mode():
    """app_companion.py의 storybook 모드에서 호출되는 화면."""
    st.markdown(
        "<p style='text-align:center; font-size:30px; font-weight:700;"
        " color:#4a7c59; margin:8px 0 4px;'>📖 내 이야기 스토리북</p>"
        "<p style='text-align:center; font-size:18px; color:#5C4033;"
        " margin:0 0 20px;'>지금까지 나눈 이야기를 한 편의 책으로 엮어드려요.</p>",
        unsafe_allow_html=True,
    )

    # 1) 스토리북 생성 트리거
    if st.button("✨ 지금까지의 대화로 스토리북 만들기", use_container_width=True):
        conv = st.session_state.get("messages", [])
        user_turns = [m for m in conv if m["role"] == "user"]
        if len(user_turns) < 2:
            st.warning("이야기를 조금 더 들려주신 다음에 만들어 볼까요? 😊")
        else:
            category = st.session_state.get("last_category", "감정긍정")
            with st.spinner("어르신의 이야기를 책으로 엮고 있어요..."):
                try:
                    st.session_state.storybook = build_storybook(conv, category)
                    st.session_state.page_idx = 0
                except Exception as e:
                    st.error(f"스토리북을 만들지 못했어요: {e}")

    # 2) 생성된 스토리북 보여주기 (페이지 넘기기)
    book = st.session_state.get("storybook")
    if book:
        idx = st.session_state.get("page_idx", 0)
        page = book[idx]

        if page["image"]:
            st.image(page["image"], use_container_width=True)
        st.markdown(
            f"<p style='font-size:22px; line-height:2.0; color:#3D2B1F;"
            f" padding:8px 4px;'>{page['text']}</p>",
            unsafe_allow_html=True,
        )
        st.caption(f"{idx + 1} / {len(book)} 쪽")

        col1, col2, _ = st.columns([1, 1, 3])
        with col1:
            if st.button("◀ 이전", disabled=(idx == 0), use_container_width=True):
                st.session_state.page_idx -= 1
                st.rerun()
        with col2:
            if st.button("다음 ▶", disabled=(idx == len(book) - 1),
                         use_container_width=True):
                st.session_state.page_idx += 1
                st.rerun()
    else:
        st.info("위 버튼을 누르면, 나눈 대화를 모아 그림이 있는 스토리북을 만들어 드려요.")