# 🌿 마음 말동무

홀로 지내는 어르신을 위한 AI 말동무 — 실제 어르신들의 구술 이야기를 근거로 공감하는 RAG 기반 챗봇입니다.

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| 🎤 음성 입력 | 마이크 녹음 → OpenAI Whisper STT |
| 🔊 음성 출력 | AI 답변 → OpenAI TTS → 자동 재생 |
| 🗂️ 감정·주제 분류 | 발화를 5개 카테고리로 분류 (감정긍정 / 감정부정 / 사물 / 장소 / 관계) |
| 🔍 구술 검색 (RAG) | FAISS 벡터 인덱스에서 유사한 실제 어르신 구술을 검색해 공감 근거로 활용 |
| 💬 스트리밍 응답 | OpenAI GPT 스트리밍으로 자연스러운 대화 흐름 제공 |
| 🧠 대화 기억 | 최근 5턴 히스토리를 유지해 문맥에 맞는 응답 생성 |

---

## 기술 스택

- **Frontend** : Streamlit + streamlit-mic-recorder
- **STT** : OpenAI Whisper
- **TTS** : OpenAI TTS
- **분류기** : scikit-learn (joblib 저장)
- **임베딩** : `jhgan/ko-sroberta-multitask`
- **벡터 DB** : FAISS
- **LLM** : GPT (gpt-5.5 → gpt-5.4 → gpt-5 순서 자동 폴백)
- **파이프라인** : LangGraph (분류 → 검색 → 생성)
- **데이터** : AI허브 '고령자 근현대 경험 기반 스토리 구술 데이터'

---

## 프로젝트 구조

```
companion/
├── app_companion.py      # Streamlit UI (음성 입출력 포함)
├── src/
│   ├── config.py         # 전역 설정 (경로, 모델명, 하이퍼파라미터)
│   ├── pipeline.py       # LangGraph 파이프라인 (분류 → 검색 → 생성)
│   ├── classifier.py     # 5카테고리 분류기
│   ├── retriever.py      # FAISS 벡터 검색
│   ├── generator.py      # OpenAI GPT 응답 생성 (스트리밍 포함)
│   └── voice.py          # STT / TTS 유틸
├── data/
│   ├── train_5cat.csv    # 학습 데이터
│   ├── valid_5cat.csv    # 검증 데이터
│   ├── clf_5cat.joblib   # 저장된 분류 모델
│   ├── corpus.faiss      # FAISS 인덱스
│   └── corpus_meta.pkl   # 코퍼스 메타데이터
├── requirements.txt
└── .env                  # OPENAI_API_KEY 설정 (git 제외)
```

---

## 시작하기

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

### 2. API 키 설정

프로젝트 루트에 `.env` 파일을 만들고 아래 내용을 입력하세요.

```
OPENAI_API_KEY=sk-...
```

### 3. 앱 실행

```bash
cd companion
streamlit run app_companion.py
```

---

## 파이프라인 흐름

```
사용자 발화 (텍스트 or 음성)
        ↓
   [분류기] 5개 카테고리 중 하나로 분류
        ↓
   [검색기] FAISS에서 유사 구술 TOP-K 검색
        ↓
   [생성기] GPT가 구술을 참고해 공감 응답 생성
        ↓
   텍스트 출력 + TTS 음성 재생
```

---

## 카테고리 안내

| 카테고리 | 예시 발화 |
|----------|-----------|
| 😊 감정긍정 | "손주가 왔다 갔는데 너무 예뻐" |
| 💙 감정부정 | "밤에 혼자 있으면 쓸쓸해" |
| 🏠 사물 | "젊었을 때 고향 밥이 제일 맛있었어" |
| 🗺️ 장소 | "오늘 공원에서 산책했더니 기분이 좋아" |
| 👥 관계 | "자식들이 바빠서 통 연락이 없네" |
