# -*- coding: utf-8 -*-
"""retriever.py — 구술 코퍼스 임베딩 + FAISS 의미 검색.

build_index(): train_5cat.csv 를 임베딩해 FAISS 인덱스로 저장 (최초 1회)
load_retriever(): 저장된 인덱스/메타/임베딩 모델을 불러옴
search(): 쿼리와 의미적으로 가까운 구술 top_k 반환
"""
import os
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from . import config


def build_index():
    """코퍼스를 임베딩하여 FAISS 인덱스로 저장한다."""
    df = pd.read_csv(config.TRAIN_CSV)
    df = df[df["text"].astype(str).str.len().between(20, 300)]
    df = df.sample(min(config.N_CORPUS, len(df)), random_state=1).reset_index(drop=True)

    model = SentenceTransformer(config.EMBED_MODEL)
    emb = model.encode(df["text"].tolist(), batch_size=64,
                       show_progress_bar=True, normalize_embeddings=True).astype("float32")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    faiss.write_index(index, config.INDEX_PATH)
    df[["text", "category", "label", "age", "sex"]].to_pickle(config.META_PATH)
    with open(config.EMBED_NAME_PATH, "w", encoding="utf-8") as f:
        f.write(config.EMBED_MODEL)
    return model, index, df


def load_retriever():
    """저장된 인덱스/메타/모델을 불러오거나, 없으면 build_index()."""
    if not os.path.exists(config.INDEX_PATH):
        print("[retriever] 인덱스가 없어 새로 생성합니다...")
        return build_index()
    index = faiss.read_index(config.INDEX_PATH)
    meta = pd.read_pickle(config.META_PATH)
    with open(config.EMBED_NAME_PATH, encoding="utf-8") as f:
        model = SentenceTransformer(f.read().strip())
    return model, index, meta


def search(model, index, meta, query: str, top_k: int = config.TOP_K):
    """쿼리와 가까운 구술 top_k를 [{score,text,label}] 형태로 반환."""
    qv = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(qv, top_k)
    hits = []
    for s, i in zip(scores[0], idxs[0]):
        row = meta.iloc[i]
        hits.append({"score": float(s), "text": row["text"], "label": row["label"]})
    return hits
