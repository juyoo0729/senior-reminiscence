# -*- coding: utf-8 -*-
"""프로젝트 전역 설정: 경로와 모델 이름을 한 곳에서 관리."""
import os

# 프로젝트 루트 (이 파일 기준 상위 폴더)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

# 데이터/모델 산출물 경로
TRAIN_CSV = os.path.join(DATA_DIR, "train_5cat.csv")
VALID_CSV = os.path.join(DATA_DIR, "valid_5cat.csv")
CLF_PATH = os.path.join(DATA_DIR, "clf_5cat.joblib")
INDEX_PATH = os.path.join(DATA_DIR, "corpus.faiss")
META_PATH = os.path.join(DATA_DIR, "corpus_meta.pkl")
EMBED_NAME_PATH = os.path.join(DATA_DIR, "embed_model.txt")

# 모델 이름
EMBED_MODEL = "jhgan/ko-sroberta-multitask"
CHAT_MODEL_PRIORITY = ["gpt-5.5", "gpt-5.4", "gpt-5"]

# 하이퍼파라미터
TOP_K = 4
N_CORPUS = 30000
EMOTION_NEG = "감정부정"
MAX_HISTORY_TURNS = 5  # OpenAI에 넘길 최대 대화 턴 수 (user+assistant 쌍)

# 50개 키워드 -> 5종 대분류
CAT_MAP = {
    '기쁘다':'감정긍정','즐겁다':'감정긍정','행복하다':'감정긍정','편안하다':'감정긍정','고맙다':'감정긍정','안심하다':'감정긍정','재미있다':'감정긍정','자랑스럽다':'감정긍정','반갑다':'감정긍정','그립다':'감정긍정','망설이다':'감정긍정','충격받다':'감정긍정',
    '미안하다':'감정부정','슬프다':'감정부정','불안하다':'감정부정','긴장되다':'감정부정','외롭다':'감정부정','후회하다':'감정부정','화나다':'감정부정','답답하다':'감정부정','지루하다':'감정부정','힘들다':'감정부정','부끄럽다':'감정부정',
    '선물':'사물','자동차':'사물','핸드폰':'사물','옷':'사물','책':'사물','음식':'사물','신문':'사물','꽃':'사물','컴퓨터':'사물',
    '산':'장소','집':'장소','식당':'장소','학교':'장소','공원':'장소','지하철':'장소','바다':'장소','동물원':'장소','병원':'장소',
    '강아지':'관계','친구':'관계','부모':'관계','아기':'관계','고양이':'관계','휴가':'관계','성공':'관계','칭찬':'관계','여행':'관계',
}
