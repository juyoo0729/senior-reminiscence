# 🌿 마음 말동무 (Senior AI Companion)

홀로 지내는 어르신을 위한 **RAG 기반 AI 말동무**입니다.
사용자의 발화를 받아 비슷한 처지의 어르신들이 실제로 남긴 구술을 근거로 검색하고,
그 정서를 바탕으로 따뜻하게 공감하는 응답을 생성합니다.

> AIFFEL DLthon 프로젝트 · 졸업 프로젝트(고립된 사람을 위한 AI 컴패니언) 프로토타입

---

## 핵심 아이디어

단순 감정 분류기는 발화에 감정 단어가 없으면 한계가 있습니다
(예: "딸이 집을 장만했어요" → 감정 단어 없이 사건만 서술).
그래서 **의미 기반 검색(RAG)** 으로 비슷한 정서의 실제 구술을 찾아 근거로 삼고,
LLM이 그 근거 위에서 공감 응답을 생성합니다. → 환각을 줄이고 공감의 구체성을 높임.

## 파이프라인

```
입력(사용자 발화)
  → ① 분류 (감정/주제 5종)        src/classifier.py
  → ② 검색 (FAISS 의미 검색)      src/retriever.py
  → ③ 생성 (근거 기반 공감 응답)   src/generator.py
  → 출력
※ ①~③을 LangGraph로 통합          src/pipeline.py
```

## 폴더 구조

```
companion/
├─ data/                 # csv, FAISS 인덱스, 분류기 (실행 시 생성)
├─ src/
│  ├─ config.py          # 경로·모델명·라벨 매핑 등 전역 설정
│  ├─ classifier.py      # 5종 분류 (KLUE/RoBERTa 우선, 없으면 TF-IDF 폴백)
│  ├─ retriever.py       # 임베딩 + FAISS 검색
│  ├─ generator.py       # OpenAI 공감 응답 생성
│  ├─ pipeline.py        # LangGraph 통합
│  ├─ evaluate.py        # 자체 지표 + LLM-judge 평가
│  └─ evaluate_ragas.py  # RAGAS 표준 RAG 평가
├─ app_companion.py      # Streamlit 웹 UI
└─ README.md
```

## 설치

```bash
pip install langgraph openai sentence-transformers faiss-cpu \
            scikit-learn pandas numpy joblib streamlit python-dotenv \
            transformers torch datasets accelerate ragas
```

API 키는 `companion/.env` 파일에 저장 (깃에 올리지 말 것):

```
OPENAI_API_KEY=sk-...
```

## 데이터 준비

`data/` 폴더에 `train_5cat.csv`, `valid_5cat.csv` 를 넣습니다.
(AI허브 '고령자 근현대 경험 기반 스토리 구술 데이터'의 라벨링 데이터에서
발화 텍스트 + 5종 대분류로 전처리한 결과)

## 실행

```bash
cd companion

# 1) 터미널에서 대화
python -m src.pipeline

# 2) 웹 UI
streamlit run app_companion.py

# 3) 평가
python -m src.evaluate
```

> 최초 실행 시 분류기 학습과 임베딩 인덱스 생성이 자동으로 수행됩니다
> (`data/` 에 결과물 저장, 이후 재사용).

## 평가 지표

| 구분 | 지표 | 설명 |
|------|------|------|
| 자체 | emotion_match | 입력 분류 ↔ 검색 구술 감정 일치율 |
| 자체 | avg_similarity | 평균 검색 유사도 |
| 자체 | retrieval_consistency | 검색 구술의 대분류 일관성 |
| LLM | empathy | 공감 정도 (1~5, LLM-as-a-judge) |
| LLM | faithfulness | 근거 충실성/환각 여부 (1~5) |

### 데이터셋 특성 메모
라벨(keyword)이 '감정'이 아니라 '이야기 주제'에 가까워,
emotion_match가 상대적으로 낮게 나오는 경향이 있습니다.
이는 모델 성능 문제가 아니라 데이터셋의 구조적 특성을 지표가 포착한 것입니다.

## 데이터 출처
AI허브 「고령자 근현대 경험 기반 스토리 구술 데이터」 (dataSetSn=71703)

## 향후 계획
- 웹 배포 (Streamlit Community Cloud)
- 대화 기록 저장 / 안전 응답(위험 발화 감지)

---

## 업데이트 기록

### 분류기 업그레이드 (KLUE/RoBERTa)
TF-IDF baseline 대비 한국어 사전학습 모델 파인튜닝으로 성능 향상.

| 분류기 | Accuracy | Macro F1 |
|--------|----------|----------|
| TF-IDF + LogReg (baseline) | 0.419 | 0.419 |
| KLUE/RoBERTa (fine-tuned)  | 0.529 | 0.527 |

- 특히 감정부정 F1: 0.411 → 0.528 (+0.117). 감정 단어가 없어도 문맥으로 분류.
- 학습: `python train_kobert.py` → `companion/data/kobert_5cat/` 저장
- `src/classifier.py`가 KLUE 모델이 있으면 자동으로 우선 사용 (없으면 TF-IDF 폴백)

### RAGAS 평가
표준 RAG 지표(faithfulness, answer_relevancy, context_precision)로 평가.
- 실행: `python -m src.evaluate_ragas`
- 자체 LLM-judge 평가는 `python -m src.evaluate`
