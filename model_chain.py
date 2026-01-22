from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (RunnableParallel,RunnablePassthrough,RunnableLambda)
from langchain_core.output_parsers import StrOutputParser



# 1. Load PDF

loader = PyPDFLoader("data/pdfs/sample.pdf")
documents = loader.load()


# 2. Split into chunks

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)


# 3. Embeddings + Vector Store

embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(chunks, embeddings)


# 4. Semantic Retriever

retriever = vector_store.as_retriever(
    search_type="mmr",               # semantic 
    search_kwargs={"k": 4, "fetch_k": 12}
)


# 5. LLM (Groq)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2
)


# 6. Prompt

prompt = PromptTemplate(
    template="""
You are an academic research assistant.

Using ONLY the context below, answer the question.
You may summarize or paraphrase, but do NOT add new information.
If the context is insufficient, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)


# 7. Helper function

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 8. RAG Chain

rag_chain = (
    RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
    )| prompt| llm| StrOutputParser()
)


# 9. Run Query

query = "Summarize the research paper"
response = rag_chain.invoke(query)

print("\n ANSWER:\n")
print(response)

