# -*- coding: utf-8 -*-
"""classifier.py — 발화의 감정/주제 5종 분류.

우선순위:
  1) companion/data/kobert_5cat/ 에 KLUE/RoBERTa 파인튜닝 모델이 있으면 그것 사용
  2) 없으면 TF-IDF + LogisticRegression (가벼운 baseline)
"""
import os
import joblib
import pandas as pd
from . import config

KOBERT_DIR = os.path.join(config.DATA_DIR, "kobert_5cat")
LABELS = ["감정긍정", "감정부정", "사물", "장소", "관계"]


# ─────────────────────────────────────────────────────
# (A) KLUE/RoBERTa 분류기
# ─────────────────────────────────────────────────────
class KoBERTClassifier:
    """파인튜닝된 KLUE/RoBERTa로 5종 분류."""

    def __init__(self, model_dir):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        # 저장된 라벨 매핑 우선, 없으면 기본값
        self.id2label = self.model.config.id2label

    def predict(self, text: str) -> str:
        enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=128).to(self.device)
        with self.torch.no_grad():
            logits = self.model(**enc).logits
        idx = int(logits.argmax(-1))
        return self.id2label[idx]


# ─────────────────────────────────────────────────────
# (B) TF-IDF baseline (폴백)
# ─────────────────────────────────────────────────────
class TfidfClassifier:
    def __init__(self, pipe):
        self.pipe = pipe

    def predict(self, text: str) -> str:
        return self.pipe.predict([text])[0]


def _train_tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    df = pd.read_csv(config.TRAIN_CSV)
    parts = [d.sample(min(len(d), 20000), random_state=42)
             for _, d in df.groupby("category")]
    df_s = pd.concat(parts).reset_index(drop=True)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=3)),
        ("clf", LogisticRegression(max_iter=500, C=5)),
    ])
    pipe.fit(df_s["text"].astype(str), df_s["category"])
    joblib.dump(pipe, config.CLF_PATH)
    return pipe


# ─────────────────────────────────────────────────────
# 통합 로더: 좋은 것부터 자동 선택
# ─────────────────────────────────────────────────────
def load_classifier():
    """KLUE 모델이 있으면 우선 사용, 없으면 TF-IDF."""
    if os.path.isdir(KOBERT_DIR) and os.path.exists(os.path.join(KOBERT_DIR, "config.json")):
        print("[classifier] KLUE/RoBERTa 모델 사용")
        return KoBERTClassifier(KOBERT_DIR)

    if os.path.exists(config.CLF_PATH):
        print("[classifier] TF-IDF 모델 사용")
        return TfidfClassifier(joblib.load(config.CLF_PATH))

    print("[classifier] 학습된 모델이 없어 TF-IDF 새로 학습...")
    return TfidfClassifier(_train_tfidf())


def predict(clf, text: str) -> str:
    """분류기 종류에 상관없이 동일하게 호출."""
    return clf.predict(text)
