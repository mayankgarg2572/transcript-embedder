import json, os
import whisper
from pathlib import Path
from src.process import VIDEOS_DIR, TRANS_DIR

model = whisper.load_model("base")

for video_path in VIDEOS_DIR.iterdir():
    if video_path.suffix.lower() not in {".mp4", ".mkv"}:
        continue

    res = model.transcribe(str(video_path), verbose=False)
    out_path = TRANS_DIR / (video_path.stem + ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print(f"Transcribed {video_path.name}")
