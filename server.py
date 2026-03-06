from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from kokoro import KPipeline

import numpy as np
import io
import re
from scipy.io.wavfile import write

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = KPipeline(lang_code="a")


def split_dialogue(text):
    parts = re.split(r'(".*?")', text)
    segments = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith('"') and part.endswith('"'):
            segments.append(("dialogue", part.strip('"')))
        else:
            segments.append(("narration", part))

    return segments


def style_text(role, text):
    if role == "dialogue":
        # add slight emphasis to dialogue
        return f"... {text} ..."
    return text


@app.get("/generate")
def generate(text: str):

    segments = split_dialogue(text)

    audio_chunks = []

    for role, segment in segments:

        styled = style_text(role, segment)

        try:
            generator = pipeline(styled, voice="af_heart")

            for _, _, audio in generator:
                audio_chunks.append(np.array(audio))

        except Exception as e:
            print("TTS failed:", e)

    if len(audio_chunks) == 0:
        return {"error": "No audio generated"}

    full_audio = np.concatenate(audio_chunks)

    buffer = io.BytesIO()
    write(buffer, 24000, full_audio)
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="audio/wav")