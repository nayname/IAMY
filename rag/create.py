from qdrant_client.grpc import VectorParams, Distance


import os
import glob
import hashlib
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# --------------------------------------------------------
# CONFIG
# --------------------------------------------------------
REPO_DIR = "/root/neutron/juno"  # <-- change this
COLLECTION_NAME = "juno_docs"
VECTOR_DIM = 1024  # depends on your embedder
CHUNK_SIZE = 500      # tokens or ~characters depending on your strategy
CHUNK_OVERLAP = 50    # ensures context continuity


# --------------------------------------------------------
# UTILS: load + clean + chunk
# --------------------------------------------------------
def load_markdown_files(repo_root):
    """Recursively load all markdown content from repo."""
    files = glob.glob(f"{repo_root}/**/*.md", recursive=True)
    files += glob.glob(f"{repo_root}/**/*.mdx", recursive=True)
    print(f"Found {len(files)} markdown files.")
    return files


def read_file(filepath):
    """Read file as plain text."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return ""


def clean_text(text):
    """Simple normalization."""
    text = text.replace("\t", " ")
    text = text.replace("\r", "")
    # remove excessive whitespace
    text = " ".join(text.split())
    return text


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start = end - overlap  # overlap
        if start < 0:
            break

    return chunks


def sha256_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------
# MAIN INGESTION FUNCTION
# --------------------------------------------------------
def ingest_docs(client, model):
    # 1. Prepare collection
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_DIM,
            distance=Distance.COSINE
        ),
    )
    print(f"Collection `{COLLECTION_NAME}` created.")

    # 2. Load files
    files = load_markdown_files(REPO_DIR)

    all_points = []
    counter = 0

    # 3. Process each file
    for file_path in files:
        raw = read_file(file_path)
        if not raw:
            continue

        cleaned = clean_text(raw)

        # skip extremely short pages
        if len(cleaned) < 30:
            continue

        # 4. chunk it
        chunks = chunk_text(cleaned)

        # 5. embed chunks
        vectors = model.encode(chunks, normalize_embeddings=True)

        # 6. build Qdrant payloads
        for chunk, vector in zip(chunks, vectors):
            point = {
                "id": counter,
                "vector": vector.tolist(),
                "payload": {
                    "text": chunk,
                    "file": os.path.relpath(file_path, REPO_DIR),
                    "hash": sha256_hash(chunk),
                },
            }
            all_points.append(point)
            counter += 1

        print(f"Ingested {file_path} → {len(chunks)} chunks.")

    # 7. Upload in batches
    BATCH_SIZE = 200
    for i in range(0, len(all_points), BATCH_SIZE):
        batch = all_points[i:i + BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        print(f"Uploaded batch {i // BATCH_SIZE + 1}/{(len(all_points) // BATCH_SIZE) + 1}")

    print(f"✅ Finished ingestion. Total chunks: {counter}")


# --------------------------------------------------------
# ENTRYPOINT
# --------------------------------------------------------
if __name__ == "__main__":
    # Qdrant client (localhost)
    client = QdrantClient("localhost", port=6333)

    # Choose your embedder (example)
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    ingest_docs(client, model)
