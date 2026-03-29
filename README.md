# 🤖 Production-Grade Multi-PDF RAG Assistant

This is a Retrieval-Augmented Generation (RAG) system built using Streamlit.

## 🚀 Features
- 📄 Multi-PDF upload
- 🔍 Hybrid search (FAISS + BM25)
- 🎯 Reranking with CrossEncoder
- 🧠 LLM-based answer generation (FLAN-T5)
- 💬 Chat interface with memory

## 🏗️ Architecture
1. PDF → Text extraction
2. Text → Chunking
3. Embeddings → FAISS index
4. Keyword search → BM25
5. Hybrid retrieval
6. Reranking (CrossEncoder)
7. Context building
8. Answer generation (LLM)

## ▶️ Run Locally

1️⃣ Clone the Repository

```
git clone https://github.com/Thashmila-Dewmini/Multi-PDF-AI-Research-Assistant.git
cd rag-assistant
```
2️⃣ Create a Virtual Environment (Recommended)
```
python -m venv venv
```
Windows:
```
venv\Scripts\activate
```
Mac/Linux:
```
source venv/bin/activate
```
3️⃣ Install Dependencies
```
pip install -r requirements.txt
```
4️⃣ Run the App
```
streamlit run app.py
```
5️⃣ Open in Browser
```
Local URL: http://localhost:8501
```
6️⃣ Use the App
* Upload one or more PDFs 📄
* Ask questions in the chat 💬
* Get answers with sources 📚

## 📌 Tech Stack
* Streamlit
* FAISS
* Sentence Transformers
* BM25
* Hugging Face Transformers

## 📷 Demo
![Demo](img\img1.png)




