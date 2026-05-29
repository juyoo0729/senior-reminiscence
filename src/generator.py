# -*- coding: utf-8 -*-
"""generator.py — 검색된 구술을 근거로 공감 응답 생성 (OpenAI)."""
from openai import OpenAI
from . import config

_client = None


def get_client():
    """OpenAI 클라이언트 (한 번만 생성)."""
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def generate(user_input: str, hits: list, category: str) -> str:
    """사용자 발화 + 검색 근거 + 분류 결과로 공감 응답을 생성한다."""
    context = "\n".join(f"- ({h['label']}) {h['text']}" for h in hits)
    system = (
        "당신은 홀로 지내는 어르신의 말동무가 되어주는 따뜻한 AI 친구입니다. "
        "상대의 감정을 먼저 공감하고, 짧고 다정한 말투로 대화하세요. "
        "조언을 늘어놓기보다 마음을 헤아리고, 더 이야기하고 싶게 질문을 한 가지 덧붙이세요."
    )
    tone = ("사용자가 힘든 감정을 표현하고 있으니 특히 조심스럽고 따뜻하게 답하세요. "
            if category == config.EMOTION_NEG else "")
    user = (
        f"{tone}[사용자의 말]\n{user_input}\n\n"
        f"[참고: 비슷한 처지의 어르신들이 실제로 하신 이야기]\n{context}\n\n"
        "위 이야기들의 정서를 참고하되 그대로 베끼지 말고, 진심으로 공감하는 답변을 2~3문장으로 해주세요."
    )
    resp = get_client().chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()
