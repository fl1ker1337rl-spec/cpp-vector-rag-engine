import glob
import os
import sys
from sentence_transformers import SentenceTransformer

try:
  from llama_cpp import Llama
except ImportError:
  print("❌ Установи llama-cpp-python: pip3 install llama-cpp-python")
  sys.exit(1)

# Определяем абсолютный путь к корню проекта
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Подключаем путь к C++ модулю из папки build
build_path = os.path.join(root_dir, "build")
sys.path.append(build_path)

import vector_engine

# Инициализируем C++ индекс и модель эмбеддингов
index = vector_engine.VectorIndex()
model = SentenceTransformer("all-MiniLM-L6-v2")

# Цикл для чтения и чанкинга файлов из папки docs (путь относительно корня)
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

# Сохраняем индекс строго в корень проекта
index_file_path = os.path.join(root_dir, "index.bin")
index.save_to_file(index_file_path)

# Загружаем модель для генерации (путь относительно корня)
model_file_path = os.path.join(root_dir, "models", "Phi-3-mini-4k-instruct-q4.gguf")
llm = Llama(model_path=model_file_path, n_ctx=2048, verbose=False)

print("\n--- RAG система готова! Задайте вопрос ---")

while True:
  query = input("\nВопрос (или 'exit' для выхода): ")
  if query.lower() == "exit":
    break

  # Конвертируем вектор запроса в list для C++ модуля
  query_vector = model.encode(query).tolist()
  top_ids = index.search(query_vector, top_k=8)

  # Собираем найденный контекст из text_storage
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

  print("\n Ответ модели:")
  print(output["choices"][0]["text"].strip())