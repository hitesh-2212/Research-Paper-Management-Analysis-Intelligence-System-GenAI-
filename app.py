import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (RunnableParallel,RunnablePassthrough,RunnableLambda)
from langchain_core.output_parsers import StrOutputParser



# CONFIG

PDF_DIR = "data/pdfs"
os.makedirs(PDF_DIR, exist_ok=True)

st.set_page_config(page_title="Research Intelligence System", layout="wide")
st.title("📚 Research Paper Intelligence System")
st.write("Upload a research paper PDF and ask semantic questions.")



# PDF UPLOAD SECTION (NEW)
# -----------------------------
st.sidebar.header("📄 Upload Research Paper")

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF file",
    type=["pdf"]
)

if uploaded_file is not None:
    file_path = os.path.join(PDF_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success(f"Uploaded: {uploaded_file.name}")
    st.cache_resource.clear()   



# BUILD RAG SYSTEM
# -----------------------------
@st.cache_resource
def build_rag_system():
    all_docs = []

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        return None, None

    for file in pdf_files:
        path = os.path.join(PDF_DIR, file)
        loader = PyPDFLoader(path)
        docs = loader.load()

        for d in docs:
            d.metadata["paper"] = file

        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEndpointEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 15}
    )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2
    )

    prompt = PromptTemplate(
        template="""
You are an academic research assistant.

Using ONLY the context below, provide a structured answer.

Context:
{context}

Question:
{question}

Answer format:
- Problem:
- Approach:
- Key Contributions:
- Limitations:
""",
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(
            f"[Paper: {d.metadata.get('paper')}, Page: {d.metadata.get('page')}] {d.page_content}"
            for d in docs
        )

    rag_chain = (
        RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


with st.spinner("Preparing research assistant..."):
    rag_chain, retriever = build_rag_system()

if rag_chain is None:
    st.warning("Please upload at least one PDF to begin.")
    st.stop()

st.success("System ready!")


# QUERY UI
# -----------------------------
query = st.text_input(
    "Ask a research question",
    placeholder="e.g. Summarize the research paper"
)

if st.button("🔍 Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer = rag_chain.invoke(query)

        st.subheader("🧠 Answer")
        st.write(answer)

        st.subheader("📄 Source Pages")
        sources = retriever.invoke(query)
        for s in sources:
            st.markdown(
                f"- **{s.metadata.get('paper')}**, page {s.metadata.get('page')}"
            )


