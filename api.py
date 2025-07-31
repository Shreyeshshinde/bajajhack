import os
import requests
import tempfile
import asyncio # <-- Add this import
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import List, Dict
from langchain_core.runnables import Runnable

from langchain_google_genai import ChatGoogleGenerativeAI
from rag_pipeline import (
    create_rag_chain,
    load_and_split_documents,
    create_vector_store,
)

load_dotenv()
app = FastAPI(title="High-Performance Q&A API", version="4.0.0")

# In-memory cache to store RAG chains for processed documents
RAG_CHAIN_CACHE: Dict[str, Runnable] = {}

class ApiQueryRequest(BaseModel):
    document_url: str = Field(..., alias="documents")
    questions: List[str]

class Answer(BaseModel):
    question: str
    answer: str

class ApiResponse(BaseModel):
    answers: List[Answer]

def get_or_create_rag_chain(doc_url: str) -> Runnable:
    """Checks cache for a RAG chain; creates and caches it if not found."""
    if doc_url in RAG_CHAIN_CACHE:
        print(f"RAG chain for {doc_url} found in cache.")
        return RAG_CHAIN_CACHE[doc_url]

    print(f"RAG chain for {doc_url} not in cache. Creating new chain...")
    temp_file_path = None
    try:
        response = requests.get(doc_url)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        chunks = load_and_split_documents(temp_file_path)
        vector_store = create_vector_store(chunks)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0)
        
        rag_chain = create_rag_chain(chunks, vector_store, llm)
        RAG_CHAIN_CACHE[doc_url] = rag_chain
        return rag_chain
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/ask", response_model=ApiResponse)
async def ask_question(request: ApiQueryRequest):
    """Processes questions concurrently using a cached or newly created RAG chain."""
    try:
        rag_chain = get_or_create_rag_chain(request.document_url)
        
        # --- THIS IS THE UPDATED ASYNCHRONOUS LOGIC ---
        # 1. Create a list of asynchronous tasks for each question
        tasks = [rag_chain.ainvoke(q) for q in request.questions]

        # 2. Run all tasks concurrently and wait for all to complete
        answers_list = await asyncio.gather(*tasks)

        # 3. Combine the original questions with their corresponding answers
        results = [
            Answer(question=q, answer=a)
            for q, a in zip(request.questions, answers_list)
        ]
        
        return ApiResponse(answers=results)
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))