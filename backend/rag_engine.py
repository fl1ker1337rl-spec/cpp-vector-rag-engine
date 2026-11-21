import glob
import os
import re
import sys
from sentence_transformers import SentenceTransformer

try:
    from llama_cpp import Llama
except ImportError:
    print("❌ Install llama-cpp-python: pip3 install llama-cpp-python")
    sys.exit(1)


def clean_book_text(raw_text: str) -> str:
    """Clean raw book text: strip table of contents header and unwrap line breaks inside paragraphs."""
    # Cut off header/TOC before CHAPTER I. if present
    chap_match = re.search(r"\bCHAPTER\s+I\b", raw_text)
    if chap_match:
        raw_text = raw_text[chap_match.start() :]

    # Normalize line endings
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    # Split into paragraphs separated by blank lines
    paragraphs = re.split(r"\n\s*\n+", text)
    cleaned_paragraphs = []

    for para in paragraphs:
        # Join single line breaks inside a paragraph into spaces and collapse extra whitespace
        single_line = re.sub(r"\s+", " ", para).strip()
        if single_line:
            cleaned_paragraphs.append(single_line)

    return "\n\n".join(cleaned_paragraphs)


# Resolve the absolute path to the project root
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the build directory path so the C++ extension module can be imported
build_path = os.path.join(root_dir, "build")
sys.path.append(build_path)

import vector_engine

# Initialize the C++ vector index and embedding model
index = vector_engine.VectorIndex()
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read, clean, and chunk .txt files from the docs directory
text_storage = {}
doc_id = 1

docs_pattern = os.path.join(root_dir, "docs", "*.txt")
for filepath in glob.glob(docs_pattern):
    with open(filepath, "r", encoding="utf-8") as f:
        content = clean_book_text(f.read())
        chunk_size = 600
        overlap = 140
        for start in range(0, len(content), chunk_size - overlap):
            chunk = content[start : start + chunk_size].strip()
            if not chunk:
                continue
            vector = model.encode(chunk).tolist()
            index.add_vector(doc_id, vector)
            text_storage[doc_id] = chunk
            doc_id += 1

# Save the index directly in the project root
index_file_path = os.path.join(root_dir, "index.bin")
index.save_to_file(index_file_path)

# Load the generation model (path relative to the project root)
model_file_path = os.path.join(root_dir, "models", "Phi-3-mini-4k-instruct-q4.gguf")
llm = Llama(model_path=model_file_path, n_ctx=2048, verbose=False)

print("\n--- RAG system is ready! Ask a question ---")

while True:
    query = input("\nQuestion (or 'exit' to quit): ")
    if query.lower() == "exit":
        break

    # Convert the query vector into a Python list for the C++ module
    query_vector = model.encode(query).tolist()
    top_ids = index.search(query_vector, top_k=5)

    # Collect matching chunks from text_storage
    retrieved_texts = []
    for idx, doc_id in enumerate(top_ids, start=1):
        if doc_id in text_storage:
            retrieved_texts.append(f"[Excerpt {idx}]\n{text_storage[doc_id]}")

    context = "\n\n".join(retrieved_texts)

    prompt = f"""<|system|>
You are a precise assistant answering questions strictly from the provided excerpts.
Rules:
1. Use ONLY facts explicitly present in the excerpts below.
2. Do NOT guess, extrapolate, or use outside knowledge.
3. If the excerpts do not clearly answer the question, reply concisely: "Not enough information in the provided context."<|end|>
<|user|>
Excerpts:
{context}

Question: {query}<|end|>
<|assistant|>
"""

    output = llm(
        prompt,
        max_tokens=220,
        temperature=0.0,
        stop=["<|end|>", "<|user|>", "<|system|>"],
    )

    print("\nModel response:")
    print(output["choices"][0]["text"].strip())