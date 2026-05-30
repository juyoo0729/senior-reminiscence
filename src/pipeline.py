# -*- coding: utf-8 -*-
"""pipeline.py — classifier/retriever/generator를 LangGraph로 통합.

입력 → 분류 → 검색 → 생성 → 출력 의 그래프를 구성한다.
build_companion()으로 컴파일된 앱을 얻고, app.invoke({"user_input": ...}) 로 실행.
build_retriever()는 분류+검색만 하는 경량 함수를 반환한다 (스트리밍 생성용).
"""
from typing import TypedDict, List, Dict, Optional

from . import classifier, retriever, generator


class State(TypedDict):
    user_input: str
    category: str
    hits: List[Dict]
    answer: str
    history: Optional[List[Dict]]


def build_companion():
    """리소스를 로드하고 LangGraph 앱을 컴파일해 반환한다."""
    from langgraph.graph import StateGraph, START, END  # 버전 충돌 방지: 필요 시점에만 임포트
    clf = classifier.load_classifier()
    embed_model, index, meta = retriever.load_retriever()

    def classify_node(state: State):
        return {"category": classifier.predict(clf, state["user_input"])}

    def retrieve_node(state: State):
        return {"hits": retriever.search(embed_model, index, meta, state["user_input"])}

    def generate_node(state: State):
        ans = generator.generate(
            state["user_input"], state["hits"], state["category"],
            history=state.get("history"),
        )
        return {"answer": ans}

    g = StateGraph(State)
    g.add_node("classify", classify_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_edge(START, "classify")
    g.add_edge("classify", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


def build_retriever():
    """분류+검색만 수행하는 함수를 반환한다 (스트리밍 생성 시 사용).

    반환 함수 시그니처: (user_input: str) -> {"category": str, "hits": list}
    """
    clf = classifier.load_classifier()
    embed_model, index, meta = retriever.load_retriever()

    def run(user_input: str) -> dict:
        cat = classifier.predict(clf, user_input)
        hits = retriever.search(embed_model, index, meta, user_input)
        return {"category": cat, "hits": hits}

    return run


if __name__ == "__main__":
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        print("[오류] OPENAI_API_KEY 없음")
        raise SystemExit(1)
    app = build_companion()
    print("준비 완료! (종료: quit)\n")
    while True:
        ui = input("나: ").strip()
        if ui.lower() in ("quit", "exit", "종료", ""):
            break
        r = app.invoke({"user_input": ui})
        print(f"\n🤖 {r['answer']}\n   [분류: {r['category']}]")
        for h in r["hits"]:
            print(f"     ({h['score']:.3f}) [{h['label']}] {h['text'][:45]}")
        print()
