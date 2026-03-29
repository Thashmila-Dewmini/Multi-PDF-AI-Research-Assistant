import streamlit as st
import numpy as np
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder  
# CrossEncoder: used for text pair classification or regression, designed to achieve maximum accuracy in measuring the similarity between two texts
from rank_bm25 import BM25Okapi
from transformers import T5Tokenizer, T5ForConditionalGeneration


st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Production-Grade Multi-PDF RAG Assistant")


@st.cache_resource   # prevents reloading every time 
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # scores pairs (query, passage) 

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

    return embed_model, reranker, tokenizer, model


embed_model, reranker, tokenizer, model = load_models()


if "messages" not in st.session_state:
    # store chat history
    st.session_state.messages = []

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "bm25" not in st.session_state:
    st.session_state.bm25 = None


def extract_text(pdf):
    reader = PdfReader(pdf)
    text = ""

    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"

    return text


def chunk_text(text, source, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + chunk_size]

        chunks.append({
            "text": chunk,
            "source": source
        })

        start += chunk_size - overlap   # overlap of 100 characters to maintain context between chunks

    return chunks



def build_indexes(chunks):
    texts = [c["text"] for c in chunks]

    # Vector index (FAISS)
    embeddings = embed_model.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    # BM25 :  traditional keyword-based retrieval method (TF-IDF)
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    return index, bm25



def retrieve(query, index, bm25, chunks, k=5):

    # Vector search: find closest chunks in vector space
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    vector_results = [chunks[i] for i in indices[0]]

    # Keyword search: score chunks based on keyword matches
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    keyword_indices = np.argsort(bm25_scores)[-k:]
    keyword_results = [chunks[i] for i in keyword_indices]

    # Combine: merge results and remove duplicates while preserving order
    combined = vector_results + keyword_results
    unique = {c["text"]: c for c in combined}

    return list(unique.values())



def rerank(query, contexts, top_k=3):
    # The CrossEncoder looks at query and chunk together making  it much more accurate

    pairs = [[query, c["text"]] for c in contexts]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(contexts, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [r[0] for r in ranked[:top_k]]



def build_context(contexts, max_chars=2500):

    text = ""

    for c in contexts:
        chunk = f"[Source: {c['source']}]\n{c['text']}\n\n"  # adds source label

        if len(text) + len(chunk) < max_chars:   # prevent prompt overflow 
            text += chunk
        else:
            break

    return text.strip()



st.sidebar.header("📂 Upload PDFs")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

# ALWAYS rebuild index when new PDFs uploaded
if uploaded_files:

    with st.spinner("Processing PDFs..."):

        all_chunks = []

        for pdf in uploaded_files:
            text = extract_text(pdf)
            chunks = chunk_text(text, pdf.name)
            all_chunks.extend(chunks)

        index, bm25 = build_indexes(all_chunks)

        st.session_state.index = index
        st.session_state.chunks = all_chunks
        st.session_state.bm25 = bm25

    st.sidebar.success("✅ Documents indexed successfully!")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


query = st.chat_input("Ask a question about your documents...")

if query and st.session_state.index:

    # Save user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

  
    contexts = retrieve(
        query,
        st.session_state.index,
        st.session_state.bm25,
        st.session_state.chunks
    )

    contexts = rerank(query, contexts, top_k=5)

    context_text = build_context(contexts)

    
    prompt = f"""
You are an expert AI research assistant.

Your task is to generate a HIGH-QUALITY, DETAILED answer.

STRICT RULES:
- Use ONLY the provided context
- Do NOT hallucinate
- If answer is not found → say "Not found in document"

ANSWER STYLE:
- Give a clear explanation (5–8 sentences)
- Explain concepts step-by-step if needed
- Include key details, definitions, and examples if available
- Make the answer easy to understand

Context:
{context_text}

Question:
{query}

Final Answer:
"""

    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=300,
        do_sample=False,
        temperature=0.4,
        top_p=0.9,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = decoded.split("Answer:")[-1].strip()

    
    with st.chat_message("assistant"):

        st.markdown(answer)

        st.markdown("### 📚 Sources")

        for i, c in enumerate(contexts):
            with st.expander(f"{c['source']} (chunk {i+1})"):
                st.write(c["text"])

    # Save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )