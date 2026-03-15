from fastapi import FastAPI, WebSocket
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


@app.websocket("/stream")
async def stream_audio(ws: WebSocket):

    await ws.accept()

    text = await ws.receive_text()

    segments = split_dialogue(text)

    for role, segment in segments:

        try:
            generator = pipeline(segment, voice="af_heart")

            for _, _, audio in generator:

                audio_np = np.array(audio)

                buffer = io.BytesIO()
                write(buffer, 24000, audio_np)
                buffer.seek(0)

                await ws.send_bytes(buffer.read())

        except Exception as e:
            print("TTS error:", e)

    await ws.close()