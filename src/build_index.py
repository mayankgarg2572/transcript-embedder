import faiss, os, numpy as np
import pickle
from src.process import EMB_DIR, CHUNKS_DIR, INDEX_DIR


emb_files = sorted(os.listdir(EMB_DIR))
d = np.load(EMB_DIR / emb_files[0], mmap_mode="r").shape[0]
index = faiss.IndexFlatIP(d)
metas = []

for fname in emb_files:
    emb = np.load(EMB_DIR / fname)
    index.add(emb.reshape(1, -1))
    vid, idx = fname.rsplit(".",1)[0].rsplit("_",1)
    times = np.load(CHUNKS_DIR / f"{vid}_{idx}.npy")
    metas.append((vid, float(times[0]), float(times[1])))


faiss.write_index(index, str(INDEX_DIR / "video.index"))
with open(INDEX_DIR / "meta.pkl", "wb") as f:
    pickle.dump(metas, f)
print("Index built:", index.ntotal)
