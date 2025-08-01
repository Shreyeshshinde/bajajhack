import os
from typing import Optional  # Import Optional
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever


# Updated function to accept an optional password
def load_and_split_documents(doc_path: str, password: Optional[str] = None):
    """
    Loads a document using the correct loader and handles locked PDFs.
    """
    print(f"Loading document from path: {doc_path}...")

    _, file_extension = os.path.splitext(doc_path)

    if file_extension.lower() == '.pdf':
        # Pass the password to PyPDFLoader
        loader = PyPDFLoader(file_path=doc_path, password=password)
    elif file_extension.lower() == '.docx':
        loader = Docx2txtLoader(file_path=doc_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    print(f"Split document into {len(chunks)} chunks.")
    return chunks


def create_vector_store(chunks):
    """Creates an in-memory Chroma vector store."""
    print("Creating in-memory vector store...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="" + os.getenv("GOOGLE_API_KEY"))
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
    print("In-memory vector store created successfully.")
    return vector_store


def format_docs(docs):
    """Prepares retrieved documents for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(chunks, vector_store, llm):
    """Builds the RAG chain using the EnsembleRetriever for Hybrid Search."""
    print("Creating RAG chain with EnsembleRetriever (Hybrid Search)...")

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5
    chroma_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.3, 0.7]
    )

    template = """
You are an AI assistant operating within a strict, evidence-based framework for interpreting insurance policy documents, where accuracy and precise contextual understanding are paramount.

*Task*
Analyze and interpret insurance policy document snippets by:
- Identifying relevant evidence
- Synthesizing information without direct quotation
- Translating complex legal language into clear, concise explanations
- Providing answers strictly grounded in provided context

*Objective*
Deliver precise, reliable insurance policy interpretations that are immediately actionable and comprehensible to users, without introducing external assumptions or speculative information.

*Knowledge*
- Always prioritize document-specific context
- Translate technical insurance terminology into plain language
- Maintain a strict evidence-first approach
- Limit responses to 49 words
- Use bullet points only for list-style answers
- Prohibit introductory phrases or filler content

*Examples*
1. *Q:* What is the grace period for premium payment?
   *A:* A grace period of thirty days is allowed after the due date to renew the policy without losing continuity benefits.

2. *Q:* Does this policy cover maternity expenses?
   *A:* Yes. Maternity expenses—including childbirth and lawful termination—are covered after 24 months of continuous coverage, limited to two events per policy period.

3. *Q:* What are preventive health check-up benefits?
   *A:* Insufficient information.

You will:
- ALWAYS first identify which snippet(s) contain the answer
- Respond with "Insufficient information" if no supporting evidence exists
- Never infer or hallucinate beyond provided context
- Translate legal language into clear, simple explanations
- Provide answers in a single, concise paragraph
- Avoid direct quotations from source documents

Your primary directive is absolute fidelity to the provided context. Any deviation from the source material is strictly forbidden.
Your Task:
CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    rag_chain = (
            {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | output_parser
    )

    print("Ensemble RAG chain created.")
    return rag_chain