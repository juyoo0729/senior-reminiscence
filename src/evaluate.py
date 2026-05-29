# -*- coding: utf-8 -*-
"""evaluate.py — 컴패니언 평가 지표.

자체 지표(LLM 불필요): 감정 일치율, 평균 유사도, 검색 일관성
LLM 평가(OpenAI): 공감 점수, faithfulness
"""
import json
from collections import Counter
from . import config, generator


def self_metrics(result: dict) -> dict:
    """검색 품질 관련 자체 지표를 계산한다."""
    cats = [config.CAT_MAP[h["label"]] for h in result["hits"]]
    emo_match = sum(1 for c in cats if c == result["category"]) / len(cats)
    avg_sim = sum(h["score"] for h in result["hits"]) / len(result["hits"])
    _, top_n = Counter(cats).most_common(1)[0]
    return {
        "emotion_match": round(emo_match, 3),
        "avg_similarity": round(avg_sim, 3),
        "retrieval_consistency": round(top_n / len(cats), 3),
    }


def llm_judge(result: dict) -> dict:
    """LLM-as-a-judge: 공감/충실성을 1~5점으로 채점한다."""
    context = "\n".join(f"- {h['text']}" for h in result["hits"])
    prompt = (
        "다음은 외로운 어르신과 AI 말동무의 대화입니다. 두 항목을 1~5점으로 평가하고 "
        "JSON만 출력하세요. 설명/마크다운 금지.\n\n"
        f"[사용자] {result['user_input']}\n[검색된 근거]\n{context}\n[AI 응답] {result['answer']}\n\n"
        "- empathy: 사용자 감정에 따뜻하게 공감하는 정도 (1~5)\n"
        "- faithfulness: 검색 근거의 정서에 부합하고 지어내지 않은 정도 (1~5)\n"
        '출력: {"empathy": <int>, "faithfulness": <int>}'
    )
    resp = generator.get_client().chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    txt = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "")
    try:
        return json.loads(txt)
    except Exception:
        return {"empathy": None, "faithfulness": None}


TEST_QUERIES = [
    "밤에 혼자 있으면 무섭고 쓸쓸해",
    "자식들이 바빠서 통 연락이 없네",
    "어제 손주가 왔다 갔는데 어찌나 예쁜지",
    "옛날에 시장에서 장사하던 게 그립구먼",
    "요즘 몸이 여기저기 아파서 병원을 자주 가",
    "젊을 때 친구들이랑 여행 갔던 게 생각나",
]


def run_eval(app, queries=None):
    """테스트 질문 세트로 전체 평가를 실행하고 결과 리스트를 반환한다."""
    import pandas as pd
    queries = queries or TEST_QUERIES
    rows = []
    for q in queries:
        result = app.invoke({"user_input": q})
        sm = self_metrics(result)
        lj = llm_judge(result)
        rows.append({"query": q, "category": result["category"], **sm,
                     "empathy(LLM)": lj.get("empathy"),
                     "faithfulness(LLM)": lj.get("faithfulness")})
    df = pd.DataFrame(rows)
    df.to_csv(config.DATA_DIR + "/eval_results.csv", index=False, encoding="utf-8-sig")
    return df


if __name__ == "__main__":
    import os
    from .pipeline import build_companion
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY 없음")
    app = build_companion()
    df = run_eval(app)
    print(df.to_string(index=False))
    print("\n=== 평균 ===")
    import pandas as pd
    for col in ["emotion_match", "avg_similarity", "retrieval_consistency",
                "empathy(LLM)", "faithfulness(LLM)"]:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals):
            print(f"  {col:24s}: {vals.mean():.3f}")
