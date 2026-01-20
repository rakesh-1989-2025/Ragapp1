import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from pinecone import Pinecone

# ---------------- ENV ---------------- #
load_dotenv(override=True)

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="📄 PDF Chatbot (Gemini + Pinecone)",
    layout="wide"
)

st.title("📄 PDF Question Answering App")
st.caption("Powered by Gemini + LangChain + Pinecone")

# ---------------- INIT MODELS (CACHE) ---------------- #

@st.cache_resource
def load_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

@st.cache_resource
def load_pinecone_index():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    return pc.Index(os.getenv("PINECONE_INDEX_NAME"))

embeddings = load_embeddings()
model = load_llm()
pinecone_index = load_pinecone_index()

# ---------------- PROMPT ---------------- #

PROMPT = PromptTemplate.from_template("""
You are a helpful assistant answering questions based on the provided documentation.

Context from the documentation:
{context}

Question: {question}

Instructions:
- Answer ONLY using the context above
- If the answer is not in the context, say:
  "I don't have enough information to answer that question."
- Be concise and clear
- Use code examples if present in the context

Answer:
""")

chain = PROMPT | model | StrOutputParser()

# ---------------- CHAT FUNCTION ---------------- #

def ask_question(question: str):
    # 1️⃣ Embed query
    query_vector = embeddings.embed_query(question)

    # 2️⃣ Pinecone search
    results = pinecone_index.query(
        vector=query_vector,
        top_k=10,
        include_metadata=True
    )

    # 3️⃣ Build context
    context = "\n\n---\n\n".join(
        match["metadata"]["text"]
        for match in results["matches"]
        if "metadata" in match and "text" in match["metadata"]
    )

    # 4️⃣ Run chain
    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response, context

# ---------------- SESSION STATE ---------------- #

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- UI ---------------- #

user_question = st.text_input(
    "Ask a question from the PDF",
    placeholder="e.g. What is Node.js event loop?"
)

ask_button = st.button("Ask")

if ask_button and user_question:
    with st.spinner("Thinking... 🤔"):
        answer, context = ask_question(user_question)

    # Save history
    st.session_state.chat_history.append(
        (user_question, answer)
    )

# ---------------- DISPLAY CHAT ---------------- #

for q, a in reversed(st.session_state.chat_history):
    st.markdown(f"**🧑 You:** {q}")
    st.markdown(f"**🤖 Bot:** {a}")
    st.divider()

# ---------------- DEBUG (OPTIONAL) ---------------- #

with st.expander("🔍 Debug: Retrieved Context"):
    if ask_button:
        st.write(context if user_question else "")
