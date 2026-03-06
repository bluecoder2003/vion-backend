from kokoro import KPipeline
import soundfile as sf

pipeline = KPipeline(lang_code="a")

text = "The wind blew softly across the forest as the sun slowly began to rise."

generator = pipeline(
    text,
    voice="af_heart"   # voice added here
)

for i, (gs, ps, audio) in enumerate(generator):
    sf.write("speech.wav", audio, 24000)

print("Audio generated!")