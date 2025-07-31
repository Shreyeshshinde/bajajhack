import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import BM25Retriever, EnsembleRetriever

def load_and_split_documents(doc_path):
    """Loads a PDF using Unstructured and splits it into chunks."""
    print(f"Loading document from path: {doc_path}...")
    # Use UnstructuredPDFLoader for better parsing of complex PDFs
    loader = UnstructuredPDFLoader(file_path=doc_path)
    documents = loader.load()
            
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    
    print(f"Split document into {len(chunks)} chunks.")
    return chunks

def create_vector_store(chunks):
    """Creates an in-memory Chroma vector store."""
    print("Creating in-memory vector store...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
    print("In-memory vector store created successfully.")
    return vector_store

def format_docs(docs):
    """Prepares retrieved documents for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(chunks, vector_store, llm):
    """Builds the RAG chain using the EnsembleRetriever for Hybrid Search."""
    print("Creating RAG chain with EnsembleRetriever (Hybrid Search)...")

    # 1. Initialize the keyword retriever (BM25)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5 # Retrieve top 5 keyword-based results

    # 2. Initialize the vector store retriever for semantic search
    chroma_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # 3. Initialize the Ensemble Retriever to combine both methods
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.5, 0.5] # Give equal weight to both semantic and keyword search
    )
    
    template = """
    You are an expert assistant for answering questions about policy documents.
    Use the following retrieved context to answer the user's question.
    Synthesize the information from all provided context snippets to form a complete and accurate answer.
    If the context does not contain the answer, state that you cannot find the information in the provided document.
    Do not use any external knowledge.

    CONTEXT: 
    {context}

    QUESTION: 
    {question}

    ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    # Define the final RAG chain
    rag_chain = (
        {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )
    
    print("Ensemble RAG chain created.")
    return rag_chain