import faiss, pickle, numpy as np
from sentence_transformers import SentenceTransformer
from process import INDEX_DIR,EMBED_MODEL, TOP_K
import sys
import time

st_model = SentenceTransformer(EMBED_MODEL)
index = faiss.read_index(str(INDEX_DIR / "video.index"))
metas = pickle.load(open(str(INDEX_DIR / "meta.pkl"), "rb"))

def embed_query(q: str):
    emb = st_model.encode(q, convert_to_numpy=True, normalize_embeddings=True)
    return emb

def search(q: str, k=TOP_K):
    q_emb = embed_query(q).reshape(1, -1)
    D, I = index.search(q_emb, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        vid, start, end = metas[idx]
        results.append({
            "video": vid,
            "start": float(start),
            "end": float(end),
            "score": float(score)
        })
    return results

if __name__ == "__main__":
    start_time = time.time()
    query = " ".join(sys.argv[1:]) or input("Query: ")
    results = search(query, k=5)
    elapsed = time.time() - start_time
    for top in results:
        if top['score'] < 0.1:
            continue
        print(f"{top['video']} Start:{top['start']:.1f}s - End:{top['end']:.1f}s (score={top['score']:.3f})")
    print(f"Search time: {elapsed:.4f} seconds")

