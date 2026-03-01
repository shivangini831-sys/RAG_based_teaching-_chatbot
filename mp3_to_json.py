import whisper
import json
import os

model = whisper.load_model("base")
audios = os.listdir("audios")

os.makedirs(".jsons", exist_ok=True)

for audio in audios:
    print(audio)

    # ✅ define default values
    number = "unknown"
    title = audio.replace(".mp3", "")

    if "_" in audio:
        parts = audio.split("_")
        number = parts[0]
        title = parts[1].replace(".mp3", "")
        print(number, title)

    result = model.transcribe(
        audio=f"audios/{audio}",
        language="hi",
        task="translate",
        word_timestamps=False
    )

    chunks = []
    for segment in result["segments"]:
        chunks.append({
            "number": number,
            "title": title,
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        })

    chunks_with_metadata = {
        "chunks": chunks,
        "text": result["text"]
    }

    with open(f".jsons/{audio}.json", "w", encoding="utf-8") as f:
        json.dump(chunks_with_metadata, f)