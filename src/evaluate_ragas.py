# -*- coding: utf-8 -*-
"""
[보너스] RAGAS 정식 라이브러리로 RAG 평가
  표준 지표: faithfulness, answer_relevancy, context_recall, context_precision

필요 패키지:
    pip install ragas datasets
    $env:OPENAI_API_KEY="your_openai_api_key_here"

실행 (companion 폴더에서):
    python -m src.evaluate_ragas

주의:
  - RAGAS는 평가에 OpenAI를 호출하므로 약간의 비용/시간이 듭니다.
  - 한국어도 동작하지만 영어 중심 설계라 점수가 보수적으로 나올 수 있습니다.
  - 설치/호환 문제가 있으면 기존 src/evaluate.py (자체 LLM-judge)를 쓰세요.
"""
import os
from .pipeline import build_companion


def run_ragas(app, queries=None):
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision

    queries = queries or [
        "밤에 혼자 있으면 무섭고 쓸쓸해",
        "자식들이 바빠서 통 연락이 없네",
        "어제 손주가 왔다 갔는데 어찌나 예쁜지",
        "옛날에 시장에서 장사하던 게 그립구먼",
    ]

    records = {"question": [], "answer": [], "contexts": []}
    for q in queries:
        result = app.invoke({"user_input": q})
        records["question"].append(q)
        records["answer"].append(result["answer"])
        records["contexts"].append([h["text"] for h in result["hits"]])

    ds = Dataset.from_dict(records)
    # context_recall은 ground_truth가 필요해 제외, 정답 불필요한 지표만 사용
    scores = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision])
    return scores


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY 없음")
    print("파이프라인 로딩...")
    app = build_companion()
    print("RAGAS 평가 시작 (OpenAI 호출, 1~2분 소요)...")
    scores = run_ragas(app)
    print("\n===== RAGAS 결과 =====")
    print(scores)
