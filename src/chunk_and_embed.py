import os, json, numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path
from src.process import TRANS_DIR, EMB_DIR, CHUNKS_DIR, EMBED_MODEL, CHUNK_SECS


st_model = SentenceTransformer(EMBED_MODEL)
from torch import set_num_threads
set_num_threads(os.cpu_count())


def chunk_segments(segments, max_secs):
    chunks, cur, t0 = [], [], segments[0]["start"]
    for seg in segments:
        cur.append(seg["text"])
        if seg["end"] - t0 >= max_secs:
            chunks.append((t0, seg["end"], " ".join(cur)))
            cur, t0 = [], seg["end"]
    if cur:
        chunks.append((t0, segments[-1]["end"], " ".join(cur)))
    return chunks





for fname in os.listdir(TRANS_DIR):
    data = json.load(open(os.path.join(TRANS_DIR, fname)))
    segs = data["segments"]
    video_id = os.path.splitext(fname)[0]
    chunked = chunk_segments(segs, CHUNK_SECS)

    for i, (start, end, text) in enumerate(tqdm(chunked, desc=video_id)):
        np.save(CHUNKS_DIR / f"{video_id}_{i}.npy",
                np.array([start, end]), allow_pickle=False)

        emb = st_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

        np.save(EMB_DIR / f"{video_id}_{i}.npy", emb)
