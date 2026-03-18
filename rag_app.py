import os
import tempfile
import streamlit as st
import warnings
from typing import List, Any

# Suppress annoying warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- CONFIG ---
CHROMA_DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Using environment variable for security
API_KEY = os.environ.get("GROQ_API_KEY", "")

if not API_KEY:
    st.warning("⚠️ Please set the GROQ_API_KEY environment variable to use the AI model.")

st.set_page_config(page_title="Fast RAG Pair-Programming", layout="wide")

@st.cache_resource
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@st.cache_resource
def get_llm() -> ChatGroq:
    """Initialize Groq via LangChain (using your gsk_ key)."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=API_KEY,
        max_tokens=1024
    )

def ingest_documents(uploaded_files: List[Any]) -> None:
    """Process uploaded files, chunk, embed, and persist to Chroma."""
    if not uploaded_files:
        return
        
    docs = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in uploaded_files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
                
            # Route by extension
            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.name.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif file.name.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                st.warning(f"Skipped unsupported file type: {file.name}")
                continue
                
            try:
                loaded_docs = loader.load()
                # Inject source filename into metadata explicitly
                for d in loaded_docs:
                    d.metadata["source_filename"] = file.name
                docs.extend(loaded_docs)
            except Exception as e:
                st.error(f"Failed to load {file.name}: {str(e)}")

    if not docs:
        return

    # Split: strict 1000 char, 200 overlap rule
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    
    # Embed & Persist
    with st.spinner(f"Embedding {len(splits)} chunks into ChromaDB..."):
        Chroma.from_documents(
            documents=splits,
            embedding=get_embeddings(),
            persist_directory=CHROMA_DB_DIR
        )
    st.success(f"Ingestion complete. Added {len(splits)} chunks to Chroma.")

def format_docs(docs: List[Any]) -> str:
    """Format retrieved documents with their source filenames."""
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source_filename', 'Unknown')}]\n{d.page_content}" 
        for d in docs
    )

def main():
    st.title("⚡ High-Speed LangChain RAG")
    
    # --- 1. Ingestion Sidebar ---
    with st.sidebar:
        st.header("1. Ingest Data")
        uploaded_files = st.file_uploader("Upload PDFs, DOCX, TXT", accept_multiple_files=True)
        if st.button("Process & Embed"):
            ingest_documents(uploaded_files)
            
    # Load Persistent Vectorstore
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=get_embeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # --- 2. Setup LCEL Chain ---
    template = """You are a highly focused AI assistant. Answer the question using ONLY the provided context.
If the answer is not contained in the context, explicitly state "The answer is not in the knowledge base."
Always cite the [Source: filename] at the end of your answer if you use information from it.

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = get_llm()
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # --- 3. Chat UI ---
    st.header("2. Query Database")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        with st.chat_message("assistant"):
            # Show retrieval scores briefly for debugging
            with st.expander("Retrieval Scores (Debug)"):
                try:
                    # similarity_search_with_relevance_scores or with_score
                    docs_with_scores = vectorstore.similarity_search_with_score(query, k=5)
                    if docs_with_scores:
                        for doc, score in docs_with_scores:
                            # Chroma returns distance metrics, lower is better usually for L2
                            st.write(f"**Distance Matrix Score: {score:.3f}** | Source: {doc.metadata.get('source_filename', 'Unknown')}")
                    else:
                        st.write("No documents retrieved.")
                except Exception as e:
                    st.write(f"Could not calculate scores or empty DB: {e}")

            # Stream response
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                for chunk in rag_chain.stream(query):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"Generation failed. Check API key or DB. Error: {str(e)}")
                full_response = "Error generating response."
                
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
