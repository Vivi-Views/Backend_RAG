import os
import datetime
import json
import hashlib
import logging
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from uuid import uuid4
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import shutil
from fastapi.middleware.cors import CORSMiddleware

# ====== Config & Security Hardening ======
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

RESPONSIBLE_AI_MANIFEST = {
    "policy": "No document or chunk processed by this pipeline will be used for LLM or embedding training unless explicitly approved by client X.",
    "timestamp": datetime.datetime.now().isoformat(),
    "enforced": True
}

with open("responsible_ai_manifest.json", "w") as f:
    json.dump(RESPONSIBLE_AI_MANIFEST, f, indent=2)

# ====== Logging Setup ======
logging.basicConfig(filename='parse_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# ====== Embedding Model (local) ======
model = SentenceTransformer('all-MiniLM-L6-v2')

# ====== Vector DB Setup (FAISS) ======
embedding_dim = 384  # For all-MiniLM
index = faiss.IndexFlatL2(embedding_dim)
chunk_metadata_store = []

# ====== Utility Functions ======
def clean_text(text):
    return text.replace('\n', ' ').replace('\xa0', ' ').strip()

def chunk_text(text, max_words=100):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def encrypt_text(text):
    return hashlib.sha256(text.encode()).hexdigest()

# ====== RBAC Layer ======
RBAC_USERS = {
    "admin": ["parse", "index", "query"],
    "viewer": ["query"]
}

def check_permission(user_role, action):
    return action in RBAC_USERS.get(user_role, [])

# ====== PDF Parsing Function ======
def parse_pdf(file_path):
    doc = fitz.open(file_path)
    raw_text = []
    metadata = {
        "file_name": os.path.basename(file_path),
        "page_count": len(doc),
        "trainable": False,
        "source_type": "pdf"
    }

    for i, page in enumerate(doc):
        text = page.get_text()
        if text:
            raw_text.append((i + 1, clean_text(text)))

    chunks = []
    for page_num, text in raw_text:
        logical_segments = text.split("\n\n")
        for segment in logical_segments:
            segment_chunks = chunk_text(segment)
            for chunk in segment_chunks:
                chunk_id = str(uuid4())
                encrypted_chunk = encrypt_text(chunk)
                chunks.append((chunk_id, chunk, page_num, encrypted_chunk))

    if chunks:
        chunk_texts = [c[1] for c in chunks]
        embeddings = model.encode(chunk_texts)
        index.add(np.array(embeddings).astype('float32'))

        for i, (chunk_id, chunk, page_num, encrypted) in enumerate(chunks):
            metadata_record = {
                "id": chunk_id,
                "file_name": metadata['file_name'],
                "page": page_num,
                "chunk": chunk,
                "trainable": False,
                "timestamp": datetime.datetime.now().isoformat(),
                "encrypted_hash": encrypted
            }
            chunk_metadata_store.append(metadata_record)
            logging.info(f"Chunk Stored: {chunk_id} | File: {metadata['file_name']} | Page: {page_num}")

    return [meta["chunk"] for meta in chunk_metadata_store[-len(chunks):]]

# ====== FastAPI Setup ======
app = FastAPI()
security = HTTPBasic()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/parse-multiple-pdfs")
def upload_multiple_pdfs(files: List[UploadFile] = File(...), credentials: HTTPBasicCredentials = Depends(security)):
    user_role = credentials.username.lower()
    if not check_permission(user_role, "parse"):
        raise HTTPException(status_code=403, detail="Access Denied.")

    total_chunks = 0
    all_top_chunks = []
    errors = []

    for file in files:
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            top_chunks = parse_pdf(temp_file_path)
            total_chunks += len(top_chunks)
            all_top_chunks.extend(top_chunks[:3])
            os.remove(temp_file_path)
        except Exception as e:
            errors.append({file.filename: str(e)})
            logging.error(f"Parsing Error in {file.filename}: {str(e)}")

    faiss.write_index(index, "faiss_index_store/my_index.faiss")

    return JSONResponse(
        content={
            "status": "success" if not errors else "partial_success",
            "total_chunks_indexed": total_chunks,
            "first_3_chunks": [f"***{chunk}***" for chunk in all_top_chunks[:3]],
            "errors": errors
        },
        status_code=200 if not errors else 207
    )
