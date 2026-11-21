# Local RAG Engine (C++ Vector Index + Python)

A local Retrieval-Augmented Generation (RAG) system combining a custom C++ vector index exposed via `pybind11` with a Python embedding and local LLM inference pipeline.

## Architecture

- **C++ Core (`pybind11`)**: Stores document vectors, performs cosine similarity top-k search, and handles saving/loading the index (`index.bin`).
- **Text Preprocessing & Embeddings (Python)**: Reads `.txt` files from `docs/`, strips leading table-of-contents/header noise before `CHAPTER I.`, unwraps line breaks inside paragraphs, splits text into overlapping chunks (`chunk_size=600`, `overlap=140`), and computes embeddings with `sentence-transformers` (`all-MiniLM-L6-v2`).
- **Local LLM Generation (`llama-cpp-python`)**: Retrieves top-k chunks (`top_k=5`) from the C++ index and generates grounded answers locally using `Phi-3-mini-4k-instruct-q4.gguf` (`temperature=0.0`) with a strict context-only prompt.

## Setup & Usage

1. **Build the C++ extension module**:
   ```bash
   cmake -S . -B build
   cmake --build build

2. **Prepare documents and model weights**:
 - Put .txt files into docs/.
 - Place Phi-3-mini-4k-instruct-q4.gguf into models/.
 

3.**Run the pipeline**:
```bash
python3 backend/rag_engine.py
```

# Screenshots
<img width="1158" height="279" alt="image" src="https://github.com/user-attachments/assets/52a6ee9b-1636-49a4-9117-6207c7e18380" />

# Planned Improvements
Add a FastAPI service layer on top of the RAG pipeline.
