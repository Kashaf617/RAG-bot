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
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
import base64
import chromadb
from chromadb.config import Settings

# --- CONFIG ---
CHROMA_DB_DIR = os.path.abspath("./chroma_db")
print(f"DEBUG: ChromaDB directory set to {CHROMA_DB_DIR}")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Load API key from Streamlit secrets
API_KEY = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title="Fast RAG Pair-Programming", layout="wide")

@st.cache_resource
def get_embeddings() -> HuggingFaceEmbeddings:
    print("DEBUG: Initializing HuggingFaceEmbeddings...")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@st.cache_resource
def get_llm() -> ChatGroq:
    """Initialize Groq via LangChain (using your gsk_ key)."""
    print("DEBUG: Initializing ChatGroq...")
    return ChatGroq(
        model="llama-3.1-8b-instant", # Switched to 8B for higher rate limits
        temperature=0.2,
        api_key=API_KEY,
        max_tokens=1024
    )

def encode_image(image_path: str) -> str:
    """Read an image and convert to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def ingest_documents(uploaded_files: List[Any]) -> None:
    """Process files, chunk text, describe images, embed, and persist."""
    if not uploaded_files:
        return
        
    docs = []
    
    # Initialize a specific Vision LLM (Llama 4 Scout is the 2026 multimodal standard)
    vision_llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.1,
        api_key=API_KEY,
        max_tokens=2048
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        for file in uploaded_files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
                
            # 1. Handle TEXT documents
            new_docs = []
            if file.name.lower().endswith(".pdf"):
                new_docs = PyPDFLoader(file_path).load()
            elif file.name.lower().endswith(".docx"):
                new_docs = Docx2txtLoader(file_path).load()
            elif file.name.lower().endswith(".txt"):
                new_docs = TextLoader(file_path).load()
                
            # 2. Handle IMAGES (Method 2: Vision API Summarization)
            elif file.name.lower().endswith((".jpg", ".jpeg", ".png")):
                with st.spinner(f"Analyzing image: {file.name}..."):
                    try:
                        b64_img = encode_image(file_path)
                        msg = vision_llm.invoke([
                            HumanMessage(
                                content=[
                                    {"type": "text", "text": "Analyze this image in meticulous detail. If it is a document, read all the text exactly as written. If it is a graph or chart, extract the numbers and trends. Be extremely thorough; this description will be used for a precise database search."},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                                ]
                            )
                        ])
                        
                        image_desc = f"[IMAGE DESCRIPTION]\n{msg.content}"
                        new_docs = [Document(page_content=image_desc)]
                        st.toast(f"✅ Successfully converted {file.name} to searchable text!")
                    except Exception as e:
                        st.error(f"Failed to analyze image {file.name}: {str(e)}")
            else:
                st.warning(f"Skipped unsupported file type: {file.name}")
                continue
                
            # Inject source filename immediately after loading this specific file
            d: Document
            for d in new_docs:
                d.metadata["source_filename"] = file.name
            docs.extend(new_docs)

    if not docs:
        st.warning("No valid text or images could be processed.")
        return

    # Split: strict 1000 char, 200 overlap rule
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    
    # Embed & Persist using explicit client to avoid tenant errors
    with st.spinner(f"Embedding {len(splits)} chunks into ChromaDB..."):
        client = chromadb.PersistentClient(
            path=CHROMA_DB_DIR,
            settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )
        Chroma.from_documents(
            documents=splits,
            embedding=get_embeddings(),
            client=client,
            collection_name="langchain"
        )
    st.success(f"Ingestion complete. Added {len(splits)} chunks to Chroma.")

def format_docs(docs: List[Any]) -> str:
    """Format retrieved documents with mandatory source headers for the LLM."""
    formatted = []
    for d in docs:
        source = d.metadata.get('source_filename', 'Unknown Source')
        content = d.page_content
        formatted.append(f"--- DOCUMENT START (Source: {source}) ---\n{content}\n--- DOCUMENT END ---")
    return "\n\n".join(formatted)

# --- SESSION STATE INITIALIZATION ---
# We do this globally to ensure it's ALWAYS available during any rerun trigger
if "messages_ui" not in st.session_state:
    st.session_state.messages_ui = []
if "messages_history" not in st.session_state:
    st.session_state.messages_history = []

def main():
    st.title("⚡ Multi-Modal RAG (Groq Vision + LangChain)")
    
    print(f"DEBUG: Attempting to connect to ChromaDB at {CHROMA_DB_DIR}...")
    # Load Persistent Vectorstore with explicit client and robust settings
    try:
        client = chromadb.PersistentClient(
            path=CHROMA_DB_DIR,
            settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )
        print("DEBUG: Successfully connected to primary ChromaDB.")
    except Exception as e:
        print(f"DEBUG: ChromaDB connection error: {e}")
        st.error(f"ChromaDB connection failed: {e}. Attempting recovery with a fresh database...")
        # Emergency fallback to a different folder if the current one is corrupted
        CHROMA_DB_DIR_FIX = os.path.abspath(f"{CHROMA_DB_DIR}_recovered")
        print(f"DEBUG: Falling back to {CHROMA_DB_DIR_FIX}")
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR_FIX)

    print("DEBUG: Initializing vectorstore...")
    vectorstore = Chroma(
        client=client,
        collection_name="langchain",
        embedding_function=get_embeddings()
    )
    print("DEBUG: Vectorstore initialized.")
    
    # --- 1. Ingestion Sidebar ---
    with st.sidebar:
        st.header("1. Ingest Data")
        uploaded_files = st.file_uploader("Upload PDFs, DOCX, TXT, or Pictures (JPG/PNG)", type=['pdf', 'docx', 'txt', 'png', 'jpg', 'jpeg'], accept_multiple_files=True)
        if st.button("Process & Embed"):
            ingest_documents(uploaded_files)
            st.rerun() # Refresh to update count
            
        # Sidebar Utilities
        st.divider()
        st.header("⚙️ Memory Settings")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages_ui = []
            st.session_state.messages_history = []
            st.rerun()
        
        st.info("💡 **Memory Active**: The AI remembers the last 5 turns of your conversation.")

        st.divider()
        st.header("📊 Database Status")
        try:
            # Get unique sources from the collection
            all_ids = vectorstore._collection.get(include=['metadatas'])
            metadatas = all_ids['metadatas']
            unique_sources = sorted(list(set([m.get("source_filename", "Unknown") for m in metadatas])))
            
            count = vectorstore._collection.count()
            st.success(f"Database contains **{count}** text chunks.")
            
            if unique_sources:
                st.write("📑 **Indexed Files:**")
                for src in unique_sources:
                    st.text(f"• {src}")
            
            if count == 0:
                st.warning("⚠️ No documents found. Please upload and process files first.")
        except Exception as e:
            st.error(f"Could not connect to database: {e}")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    
    # --- 2. Setup LCEL Chain with Memory ---
    template = """You are a helpful, extremely polite, and knowledgeable AI assistant. Your goal is to provide the best possible answers using the provided Context.

BEHAVIOR RULES:
1. GREETINGS: If the user says hello (e.g., 'hi', 'hy', 'assalamu alaikum'), respond with a warm greeting and ask how you can help. DO NOT mention the Context or uploaded documents.
2. FAREWELLS: If the user says goodbye (e.g., 'bye', 'allah hafiz', 'goodbye'), respond with a polite farewell (e.g., 'Allah hafiz', 'Goodbye, take care!'). DO NOT say hello, and DO NOT mention the Context.
3. CONVERSATIONAL CHAT: For "thanks", "okay", or small talk, just acknowledge politely. DO NOT mention the Context.
4. ANSWERING QUESTIONS: When asked a factual question, use the provided Context. Synthesize whatever related info you find politely.
5. INCOMPLETE INFO: If the Context is completely irrelevant to the factual question, explain: "I couldn't find the exact answer in the documents, but here is what the documents mention..." instead of a blunt denial.
6. MANDATORY CITATION: Always cite sources by appending [Source: filename] to facts you pull from the Context.
7. NO HALLUCINATION: Be helpful, but never invent facts not found in the Context.

Conversation History:
{chat_history}

Context Information:
{context}

Question: {question}

Final Answer (Always cite sources if context is used):"""
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = get_llm()
    
    def format_history(messages: List[Any]) -> str:
        return "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {getattr(m, 'content', '')}" for m in messages])

    # The Chain: Now includes chat_history with defensive access
    def get_context(query):
        docs = retriever.invoke(query)
        print(f"DEBUG: Retrieved {len(docs)} documents for query: '{query}'")
        for i, d in enumerate(docs[:3]):
            print(f"  - Chunk {i+1} Source: {d.metadata.get('source_filename', 'Unknown')}")
            print(f"    Content Snippet: {d.page_content[:100]}...")
        return format_docs(docs)

    rag_chain: Any = (
        {
            "context": get_context, 
            "question": RunnablePassthrough(),
            "chat_history": lambda x: format_history(st.session_state.get("messages_history", [])[-10:]) 
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # --- 3. Chat UI ---
    st.header("2. Search & Query")

    # Display Chat History
    for msg in st.session_state.messages_ui:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("Ask a question about your documents..."):
        # Add to UI and History
        st.session_state.messages_ui.append({"role": "user", "content": query})
        st.session_state.messages_history.append(HumanMessage(content=query))
        
        with st.chat_message("user"):
            st.markdown(query)
            
        with st.chat_message("assistant"):
            with st.expander("Retrieval Scores (Debug)"):
                try:
                    docs_with_scores = vectorstore.similarity_search_with_score(query, k=8)
                    if docs_with_scores:
                        for doc, score in docs_with_scores:
                            st.write(f"**Distance Matrix Score: {score:.3f}** | Source: {doc.metadata.get('source_filename', 'Unknown')}")
                    else:
                        st.write("No documents retrieved.")
                except Exception as e:
                    st.write(f"Could not calculate scores: {e}")

            # Stream response
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                # Execute the chain with the query
                for chunk in rag_chain.stream(query):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"Generation failed. Error: {str(e)}")
                full_response = "Error generating response."
                
        # Save AI response to history
        st.session_state.messages_ui.append({"role": "assistant", "content": full_response})
        st.session_state.messages_history.append(AIMessage(content=full_response))

if __name__ == "__main__":
    main()
