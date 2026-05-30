FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지 (faiss, torch 등에 필요할 수 있는 것들)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 의존성 먼저 설치 (캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 코드 + 데이터 복사
COPY . .

# 허깅페이스가 모델/캐시를 쓸 수 있도록 캐시 경로 지정
ENV HF_HOME=/app/.cache
ENV TRANSFORMERS_CACHE=/app/.cache

EXPOSE 8501

# Streamlit 실행 (app_companion.py, 포트 8501 고정)
CMD ["streamlit", "run", "app_companion.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false"]
