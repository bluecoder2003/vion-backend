from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from kokoro import KPipeline

import numpy as np
import io
from scipy.io.wavfile import write

import spacy
from transformers import pipeline as hf_pipeline

app = FastAPI()

app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_methods=["*"],
allow_headers=["*"],
)

# -----------------------------

# Load NLP models

# -----------------------------

nlp = spacy.load("en_core_web_sm")

emotion_classifier = hf_pipeline(
"text-classification",
model="j-hartmann/emotion-english-distilroberta-base",
)

# -----------------------------

# Kokoro TTS

# -----------------------------

pipeline = KPipeline(lang_code="a")

# -----------------------------

# Emotion → Voice Mapping

# -----------------------------

emotion_voice_map = {
"joy": "af_bella",
"anger": "af_nicole",
"sadness": "af_heart",
"fear": "af_heart",
"surprise": "af_bella",
"neutral": "af_heart"
}
# -----------------------------

# Sentence Splitting

# -----------------------------

def split_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

# -----------------------------

# Emotion Detection

# -----------------------------

def detect_emotion(text):
    try:
        result = emotion_classifier(text)
        label = result[0]["label"].lower()
        return label
    except Exception:
        return "neutral"

# -----------------------------

# WebSocket Streaming Endpoint

# -----------------------------

@app.websocket("/stream")
async def stream_audio(ws: WebSocket):
    await ws.accept()

    text = await ws.receive_text()
    sentences = split_sentences(text)

    for sentence in sentences:
        try:
            emotion = detect_emotion(sentence)
            voice = emotion_voice_map.get(emotion, "af_heart")

            print("Sentence:", sentence)
            print("Emotion:", emotion)
            print("Voice:", voice)

            generator = pipeline(sentence, voice=voice)

            for _, _, audio in generator:
                audio_np = np.array(audio)
                buffer = io.BytesIO()
                write(buffer, 24000, audio_np)
                buffer.seek(0)
                await ws.send_bytes(buffer.read())
        except Exception as e:
            print("TTS error:", e)

    await ws.close()
