---
title: Senior Reminiscence
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: streamlit
app_file: app_companion.py
pinned: false
---

# 🌿 마음 말동무

홀로 지내는 어르신을 위한 AI 말동무 — 실제 어르신들의 구술 이야기를 근거로 공감하는 RAG 기반 챗봇입니다.

🚀 **[허깅페이스 데모 바로가기](https://huggingface.co/spaces/midi3008/senior-reminiscence)**

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| 🎤 음성 입력 | 마이크 녹음 → OpenAI Whisper STT |
| 🔊 음성 출력 | AI 답변 → OpenAI TTS → 자동 재생 |
| 💬 예시 버튼 | 첫 화면에 큰 버튼으로 표시 — 누르면 바로 입력 처리 (모바일 친화) |
| 🗂️ 감정·주제 분류 | 발화를 5개 카테고리로 분류 (감정긍정 / 감정부정 / 사물 / 장소 / 관계) |
| 🔍 구술 검색 (RAG) | FAISS 벡터 인덱스에서 유사한 실제 어르신 구술을 검색해 공감 근거로 활용 |
| 💬 스트리밍 응답 | OpenAI GPT 스트리밍으로 자연스러운 대화 흐름 제공 |
| 🧠 대화 기억 | 최근 5턴 히스토리를 유지해 문맥에 맞는 응답 생성 |
| 📖 스토리북 | 나눈 대화를 GPT가 1인칭 회상 이야기로 정리 → 카테고리별 이미지와 함께 책처럼 표시 |

---

## 앱 모드

앱은 두 가지 모드를 제공합니다.

| 모드 | 설명 |
|------|------|
| 💬 말동무와 대화 | 음성·텍스트로 AI와 자유롭게 대화 |
| 📖 내 이야기 스토리북 | 지금까지 나눈 대화를 GPT가 회상 이야기로 엮어 페이지별로 감상 |

---

## 입력 방식

대화는 세 가지 방법으로 시작할 수 있습니다.

1. **예시 버튼** — 첫 화면에 표시된 버튼을 누르면 바로 전송
2. **마이크** — 🎤 버튼을 눌러 말하면 Whisper가 텍스트로 변환
3. **직접 입력** — 텍스트 입력칸에 적고 "📤 말 보내기" 버튼 또는 Enter

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
- **스토리북** : GPT 이야기 생성 + 카테고리별 저장 이미지 매칭
- **데이터** : AI허브 '고령자 근현대 경험 기반 스토리 구술 데이터'

---

## 프로젝트 구조

```
companion/
├── app_companion.py      # Streamlit UI (모드 선택 / 음성 입출력 / 모바일 친화)
├── storybook.py          # 스토리북 모드 (GPT 이야기 생성 + 이미지 매칭 + 페이지 UI)
├── src/
│   ├── config.py         # 전역 설정 (경로, 모델명, 하이퍼파라미터)
│   ├── pipeline.py       # LangGraph 파이프라인 (분류 → 검색 → 생성)
│   ├── classifier.py     # 5카테고리 분류기
│   ├── retriever.py      # FAISS 벡터 검색
│   ├── generator.py      # OpenAI GPT 응답 생성 (스트리밍 포함)
│   └── voice.py          # STT / TTS 유틸
├── images/               # 스토리북용 카테고리별 이미지 폴더
│   ├── positive/         # 감정긍정
│   ├── negative/         # 감정부정
│   ├── object/           # 사물
│   ├── place/            # 장소
│   └── relation/         # 관계
├── data/
│   ├── clf_5cat.joblib   # 저장된 분류 모델
│   ├── corpus.faiss      # FAISS 인덱스
│   ├── corpus_meta.pkl   # 코퍼스 메타데이터
│   └── embed_model.txt   # 임베딩 모델 이름
├── .env.example          # 환경변수 템플릿
├── requirements.txt
└── .env                  # OPENAI_API_KEY 설정 (git/업로드 제외)
```

---

## 시작하기

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

> **주의**: `corpus_meta.pkl`은 numpy 2.x 환경에서 생성된 파일입니다.  
> 로컬에서 `numpy._core` 오류가 발생하면 `numpy>=2.0` 으로 업그레이드하고,  
> `data/corpus.faiss`와 `data/corpus_meta.pkl`을 삭제한 뒤 앱을 실행하면 자동 재생성됩니다.

### 2. API 키 설정

프로젝트 루트에 `.env` 파일을 만들고 아래 내용을 입력하세요.

```
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. 앱 실행

```bash
cd companion
streamlit run app_companion.py
```

---

## 파이프라인 흐름

### 말동무 대화 모드

```
사용자 발화 (예시 버튼 / 마이크 / 텍스트 직접 입력)
        ↓
   [분류기] 5개 카테고리 중 하나로 분류
        ↓
   [검색기] FAISS에서 유사 구술 TOP-K 검색
        ↓
   [생성기] GPT가 구술을 참고해 공감 응답 생성 (스트리밍)
        ↓
   텍스트 출력 + TTS 음성 자동 재생
```

### 스토리북 모드

```
대화 기록 (어르신 발화 전체)
        ↓
   [GPT] 1인칭 회상 이야기 3~5 문단 생성
        ↓
   [코드] 문단 단위 페이지 분할 (결정론적)
        ↓
   [이미지] 마지막 분류 카테고리 → images/ 폴더에서 이미지 선택
        ↓
   페이지 넘기기 UI로 출력
```

---

## 카테고리 안내

| 카테고리 | 예시 발화 |
|----------|-----------|
| 😊 감정긍정 | "어제 손주가 왔다 갔는데 너무 예뻐" |
| 💙 감정부정 | "밤에 혼자 있으면 무섭고 쓸쓸해" |
| 🏠 사물 | "젊었을 때 고향 밥이 제일 맛있었어" |
| 🗺️ 장소 | "오늘 공원에서 산책했더니 기분이 좋아" |
| 👥 관계 | "자식들이 바빠서 통 연락이 없네" |
