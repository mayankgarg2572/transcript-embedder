# Text based Search Command Line Tool for Videos

This project takes video files, transcribes them, then splits them into fixed-length time chunks, and computes normalized sentence embeddings for each chunk using a local SentenceTransformer model.

---

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites) 
3. [Installation](#installation)
4. [Usage](#usage)
5. [License](#license)

---

## Features

- **Video transcription** using `whisper`
- **Automatic chunking** of transcript segments into N-second windows
- **Local embedding** with `all-MiniLM-L6-v2` (384-dim vectors, cosine-normalized)
- **Binary serialization** of chunk boundaries and embeddings via NumPy (`.npy`)
- **Chunking Progress** reporting via `tqdm`

---

## Prerequisites

- Python 3.10.0
- ffmpeg installed on your system (for video processing)

---

## Installation

1. Clone this repository:

```bash
git clone https://github.com/mayankgarg2572/transcript-embedder.git
cd transcript-embedder
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Usage
Place your video files in the videos folder. Then run the following commands:

Run the transcript extraction script:

```bash
python src/transcribe.py
```

Run the chunking and embedding script:

```bash
python src/chunk_and_embed.py
```

Run the indexing script:

```bash
python src/build_index.py
```
Run the query script to search for relevant video segments:

```bash
python src/query.py "your search query here"
```

---

# License
This project is released under the MIT License.