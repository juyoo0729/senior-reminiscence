# -*- coding: utf-8 -*-
"""voice.py — 음성 입출력 (OpenAI Whisper STT + OpenAI TTS).

- speech_to_text(audio_bytes): 음성(bytes) → 텍스트
- text_to_speech(text): 텍스트 → 음성(bytes, mp3)

RAG 본체(generator/retriever)는 건드리지 않는다. 입출구만 담당.
"""
import io
from openai import OpenAI

_client = None


def get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def speech_to_text(audio_bytes: bytes) -> str:
    """어르신 음성(wav/webm bytes) → 텍스트로 변환 (Whisper)."""
    client = get_client()
    # OpenAI SDK는 파일 객체를 받음. 이름(.wav)을 줘야 포맷을 인식한다.
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "speech.wav"

    resp = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="ko",  # 한국어 고정 → 인식 정확도 ↑
    )
    return resp.text.strip()


def text_to_speech(text: str, voice: str = "nova") -> bytes:
    """말동무 답변 텍스트 → 음성(mp3 bytes) 변환 (OpenAI TTS).

    voice 옵션: alloy, echo, fable, onyx, nova, shimmer
    어르신용으로는 부드러운 'nova' 또는 'shimmer' 추천.
    """
    client = get_client()
    resp = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
    )
    return resp.content  # mp3 bytes
