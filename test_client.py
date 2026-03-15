import asyncio
import io
from pathlib import Path

import numpy as np
import websockets
from scipy.io.wavfile import read, write


WS_URL = "ws://127.0.0.1:8000/stream"
SAMPLE_TEXT = "I can't believe you did this! She whispered sadly, I trusted you. Suddenly the door slammed open. Run! Run for your life!"
OUTPUT_FILE = Path("streamed_audio.wav")
PAUSE_MS = 180


def combine_wav_chunks(chunks: list[bytes], output_file: Path) -> None:
    sample_rate = None
    dtype = None
    combined_audio = []

    for index, chunk in enumerate(chunks):
        current_rate, audio = read(io.BytesIO(chunk))

        if sample_rate is None:
            sample_rate = current_rate
            dtype = audio.dtype
        elif current_rate != sample_rate:
            raise ValueError(
                f"Incompatible sample rate in chunk {index}: {current_rate} != {sample_rate}"
            )

        if audio.dtype != dtype:
            audio = audio.astype(dtype)

        combined_audio.append(audio)

        if index < len(chunks) - 1:
            silence_frames = int(sample_rate * (PAUSE_MS / 1000))
            silence_shape = (silence_frames,) if audio.ndim == 1 else (silence_frames, audio.shape[1])
            combined_audio.append(np.zeros(silence_shape, dtype=dtype))

    if sample_rate is None or dtype is None:
        raise ValueError("No audio chunks were received.")

    final_audio = np.concatenate(combined_audio, axis=0)
    write(output_file, sample_rate, final_audio)


async def main() -> None:
    audio_chunks = []

    async with websockets.connect(WS_URL, max_size=None) as websocket:
        await websocket.send(SAMPLE_TEXT)

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    audio_chunks.append(message)
        except websockets.ConnectionClosed:
            pass

    if not audio_chunks:
        print("No audio received from the server.")
        return

    combine_wav_chunks(audio_chunks, OUTPUT_FILE)
    print(f"Received {len(audio_chunks)} audio chunk(s).")
    print(f"Saved audio to {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
