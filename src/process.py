from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMB_DIR    = ROOT / "embeddings"
INDEX_DIR  = ROOT / "index"
CHUNKS_DIR = ROOT / "chunks"
TRANS_DIR = ROOT / "transcripts"
VIDEOS_DIR = ROOT / "videos"

# Lets first create all the directories if they don't exist
EMB_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
TRANS_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
CHUNK_SECS = 10