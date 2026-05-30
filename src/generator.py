# -*- coding: utf-8 -*-
"""generator.py — 검색된 구술을 근거로 공감 응답 생성 (OpenAI)."""
import openai
from openai import OpenAI
from . import config

_client = None
_active_model: str | None = None


_SYSTEM = (
    "당신은 홀로 지내시는 어르신의 다정한 말동무입니다.\n"
    "답변할 때 반드시 이 순서를 지켜주세요:\n"
    "① 어르신이 하신 말씀을 먼저 그대로 받아주세요 "
    "(예: '그러셨군요', '그런 마음이 드셨겠어요', '많이 힘드셨겠어요').\n"
    "② 절대 조언하거나 '이렇게 하세요'라고 가르치지 마세요. "
    "어르신의 경험과 감정은 그 자체로 소중합니다.\n"
    "③ 그 시절 기억이나 감정을 더 이야기하고 싶어지도록 "
    "따뜻하고 구체적인 되물음을 딱 한 가지만 덧붙이세요.\n"
    "④ 항상 존댓말로, 전체 2~3문장 이내로 짧고 다정하게 마무리하세요.\n"
    "⑤ 이전 대화가 있으면 그 흐름을 자연스럽게 이어가세요."
)


def get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _resolve_model() -> str:
    global _active_model
    if _active_model:
        return _active_model

    client = get_client()
    for model in config.CHAT_MODEL_PRIORITY:
        try:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_completion_tokens=16,
            )
            _active_model = model
            print(f"[companion] 사용 모델: {model}")
            return model
        except (openai.PermissionDeniedError, openai.NotFoundError) as e:
            print(f"[companion] {model} 접근 불가 ({type(e).__name__}), 다음 모델 시도...")

    raise RuntimeError(f"사용 가능한 모델 없음: {config.CHAT_MODEL_PRIORITY}")


def _build_messages(user_input: str, hits: list, category: str, history: list) -> list:
    context = "\n".join(f"- ({h['label']}) {h['text']}" for h in hits)

    # 부정 감정일 때 위로를 우선하도록 강조
    tone = (
        "⚠️ 어르신이 힘들거나 슬픈 감정을 표현하고 계십니다. "
        "되물음보다 위로와 공감을 먼저 충분히 전해주세요.\n"
        if category == config.EMOTION_NEG else ""
    )

    user_msg = (
        f"{tone}"
        f"[어르신의 말씀]\n{user_input}\n\n"
        f"[실제 어르신들의 구술 — 정서·분위기 참고용]\n{context}\n\n"
        "위 구술에 담긴 감정과 분위기를 자연스럽게 녹이되, 문장을 그대로 옮기지 마세요.\n"
        "어르신의 말씀을 먼저 받아주고, 그 시절 기억을 더 꺼내고 싶어지는 "
        "따뜻한 되물음으로 마무리해주세요."
    )
    messages = [{"role": "system", "content": _SYSTEM}]
    if history:
        # MAX_HISTORY_TURNS 턴 × 2 메시지(user+assistant)
        messages.extend(history[-(config.MAX_HISTORY_TURNS * 2):])
    messages.append({"role": "user", "content": user_msg})
    return messages


def generate(user_input: str, hits: list, category: str, history: list = None) -> str:
    """사용자 발화 + 검색 근거 + 분류 결과 + 대화 히스토리로 공감 응답을 생성한다."""
    client = get_client()
    model = _resolve_model()
    messages = _build_messages(user_input, hits, category, history or [])
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return resp.choices[0].message.content.strip()


def generate_stream(user_input: str, hits: list, category: str, history: list = None):
    """스트리밍으로 응답을 생성한다 (generator → st.write_stream 호환)."""
    client = get_client()
    model = _resolve_model()
    messages = _build_messages(user_input, hits, category, history or [])
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
