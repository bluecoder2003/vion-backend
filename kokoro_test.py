from kokoro import KPipeline
import soundfile as sf
import numpy as np

pipeline = KPipeline(lang_code="a")

text = """
I can't believe you did this!
She started crying quietly.
Suddenly the door slammed open.
The wind blew softly across the forest as the sun slowly began to rise.
"""

generator = pipeline(
text,
voice="af_heart"
)

audio_chunks = []

for i, (gs, ps, audio) in enumerate(generator):

```
print(f"Chunk {i}")
print("Graphemes:", gs)
print("Phonemes:", ps)

audio_chunks.append(audio)
```

# combine chunks

final_audio = np.concatenate(audio_chunks)

sf.write("speech.wav", final_audio, 24000)

print("Audio generated successfully!")
