import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

st.set_page_config(page_title="Smart Meeting Minutes - FYP Assistant", page_icon="🎓")
st.title("🎓 Smart Meeting Minutes — FYP Assistant")
st.write("Ask me anything about the Smart Meeting Minutes FYP project!")

@st.cache_resource
def setup_rag():
    full_text = ""
    with pdfplumber.open("SmartMinutes_FYP_FINAL.pdf") as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    words = full_text.split()
    chunks = []
    chunk_size, overlap = 500, 50
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap

    model = SentenceTransformer('all-MiniLM-L6-v2')
    chunk_embeddings = model.encode(chunks)

    client = chromadb.Client()
    collection = client.create_collection(name="fyp_docs")
    collection.add(
        documents=chunks,
        embeddings=chunk_embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    return model, collection

model, collection = setup_rag()

groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def ask_question(question, n_results=3):
    question_embedding = model.encode([question]).tolist()
    results = collection.query(query_embeddings=question_embedding, n_results=n_results)
    relevant_chunks = results['documents'][0]
    context = "\n\n".join(relevant_chunks)

    prompt = f"""You are answering questions about the "Smart Meeting Minutes" FYP project.
The context below is taken from the FYP report. Answer clearly and professionally in English, based only on this context.
If the answer is not in the context, say "This information is not available in the document."

Context:
{context}

Question: {question}

Answer (in English):"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_question = st.chat_input("Ask a question about the FYP...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_question(user_question)
            st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
