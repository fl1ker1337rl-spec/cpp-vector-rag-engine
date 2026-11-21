import glob
import os
import sys
from sentence_transformers import SentenceTransformer

try:
    from llama_cpp import Llama
except ImportError:
    print("❌ Install llama-cpp-python: pip3 install llama-cpp-python")
    sys.exit(1)

# Resolve the absolute path to the project root
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the build directory path so the C++ extension module can be imported
build_path = os.path.join(root_dir, "build")
sys.path.append(build_path)

import vector_engine

# Initialize the C++ vector index and embedding model
index = vector_engine.VectorIndex()
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read and chunk .txt files from the docs directory (relative to the project root)
text_storage = {}
doc_id = 1

docs_pattern = os.path.join(root_dir, "docs", "*.txt")
for filepath in glob.glob(docs_pattern):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        chunk_size = 800
        overlap = 250
        for start in range(0, len(content), chunk_size - overlap):
            chunk = content[start : start + chunk_size]
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
    top_ids = index.search(query_vector, top_k=8)

    # Collect matching chunks from text_storage
    retrieved_texts = []
    for doc_id in top_ids:
        if doc_id in text_storage:
            retrieved_texts.append(text_storage[doc_id])

    context = "\n".join(retrieved_texts)

    prompt = f"""<|system|>
  You are a helpful AI assistant. Answer the user's question strictly based on the provided context below. 
  If the answer cannot be found in the context, explicitly state that there is not enough information. Do not make assumptions or use outside knowledge.<|end|>
  <|user|>
  Context:
  {context}

  Question: {query}<|end|>
  <|assistant|>
  """

    output = llm(
        prompt,
        max_tokens=160,
        temperature=0.1,
        stop=["<|end|>", "<|user|>", "<|system|>"],
    )

    print("\nModel response:")
    print(output["choices"][0]["text"].strip())
