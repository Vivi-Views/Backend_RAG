<<<<<<< Updated upstream
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
=======
# Imports
# placed at their code blocks
>>>>>>> Stashed changes

#===========================================================XXX===========================================================#

# imports
from fastapi import FastAPI, Request

#-----------------------

<<<<<<< Updated upstream
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
=======
# Initialization of FastAPI
>>>>>>> Stashed changes
app = FastAPI()

<<<<<<< Updated upstream
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
=======
#===========================================================XXX===========================================================#
>>>>>>> Stashed changes

# Imports
import logging

<<<<<<< Updated upstream
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
=======
#-----------------------

# ====== Enhanced Logging Setup ======
logging.basicConfig(
    filename='rag_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(action)s | %(file_name)s | %(chunk_id)s | %(message)s'
)

# Custom Logger Adapter to inject dynamic context
class ContextualLogger(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        context = self.extra.copy()
        context.update(kwargs.get("extra", {}))
        kwargs["extra"] = context
        return msg, kwargs

logger = ContextualLogger(logging.getLogger(__name__), {
    "action": "INIT",
    "file_name": "N/A",
    "chunk_id": "N/A"
})

#===========================================================XXX===========================================================#

# Imports
from collections import defaultdict

#-----------------------

# Note:
"""
    Counts file types and logs each identification.

    Parameters:
        file_list (List[str]): List of filenames or table names.
        logger (ContextualLogger): The enhanced contextual logger.

    Returns:
        Dict[str, int]: Dictionary with file type as key and count as value.
    """


# ====== File Type Counting Function ======
def count_file_types(file_list):

    type_counts = defaultdict(int)

    logger.info("Starting file type identification", extra={
        "action": "IDENTIFY_TYPES",
        "file_name": "input_list",
        "chunk_id": "N/A"
    })

    for item in file_list:
        lower_item = item.lower()
        if lower_item.endswith('.pdf'):
            doc_type = 'pdf'
        elif lower_item.endswith('.docx'):
            doc_type = 'docx'
        elif lower_item.endswith('.xlsx'):
            doc_type = 'xlsx'
        elif lower_item.endswith('.txt'):
            doc_type = 'txt'
        elif '.' not in lower_item:
            doc_type = 'snowflake_table_or_view'
        else:
            doc_type = 'unknown'

        type_counts[doc_type] += 1

        logger.info(f"Identified type: {doc_type}", extra={
            "action": "TYPE_IDENTIFIED",
            "file_name": item,
            "chunk_id": "N/A"
        })

    logger.info(f"Type summary: {dict(type_counts)}", extra={
        "action": "TYPE_COUNT_SUMMARY",
        "file_name": "input_list",
        "chunk_id": "N/A"
    })

    return dict(type_counts)

#===========================================================XXX===========================================================#

# Imports
from pydantic import BaseModel
from typing import List

#-----------------------

class FileListRequest(BaseModel):
    files: List[str]

"""
# For passing as list from terminal using curl command
@app.post("/file-types")
def get_file_types(files: List[str]):
    return count_file_types(files)

# use like this - using curl command
curl -X POST http://127.0.0.1:8000/file-types -H "Content-Type: application/json" -d "[\"doc1.pdf\", \"notes.txt\", \"orders_table\", \"report.xlsx\", \"weird.abc\"]"

"""

# For webpage
@app.post("/file-types")
def get_file_types(request: FileListRequest):
    return count_file_types(request.files)

"""
# use like this - using curl command
curl -X POST http://127.0.0.1:8000/file-types -H "Content-Type: application/json" -d "{\"files\": [\"doc1.pdf\", \"notes.txt\", \"orders_table\"]}"
"""

#===========================================================XXX===========================================================#



#===========================================================XXX===========================================================#

# Testing from terminal
# curl http://127.0.0.1:8000/file-types

"""
@app.get("/file-types")
def get_file_types():
    # Example static call — customize as needed
    sample_input = ['doc1.pdf', 'notes.txt', 'orders_table','doc2.pdf', 'notes1.txt', 'table1', 'report.xlsx', 'weird.abc']
    return count_file_types(sample_input, logger)
"""
#===========================================================XXX===========================================================#

# Testing from terminal
# python main.py "['doc1.pdf', 'notes.txt', 'table1', 'report.xlsx', 'weird.abc']"
# python main.py "['doc1.txt']"
# python main.py : Usage: python main.py "['file1.pdf', 'table_name', 'data.xlsx']"


# ====== Main Entrypoint ======
if __name__ == "__main__":
    import sys
    import ast

    if len(sys.argv) < 2:
        print("Usage: python main.py \"['file1.pdf', 'table_name', 'data.xlsx']\"")
        sys.exit(1)

    try:
        input_list = ast.literal_eval(sys.argv[1])
        if not isinstance(input_list, list):
            raise ValueError("Input is not a list.")
    except Exception as e:
        print(f"Error: Invalid input list format. Must be a valid Python list.\nDetails: {e}")
        sys.exit(1)

    result = count_file_types(input_list, logger)

    print("\nFile Type Count Summary:")
    for file_type, count in result.items():
        print(f"{file_type} : {count}")
>>>>>>> Stashed changes
