import json, os
import whisper
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = ROOT / "videos"
TRANS_DIR  = ROOT / "transcripts"
EMB_DIR    = ROOT / "embeddings"
INDEX_DIR  = ROOT / "index"
CHUNKS_DIR = ROOT / "chunks"


model = whisper.load_model("base")

for video_path in VIDEOS_DIR.iterdir():
    if video_path.suffix.lower() not in {".mp4", ".mkv"}:
        continue

    res = model.transcribe(str(video_path), verbose=False)
    out_path = TRANS_DIR / (video_path.stem + ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print(f"Transcribed {video_path.name}")
