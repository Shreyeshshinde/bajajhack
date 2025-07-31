import os
import requests
import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from rag_pipeline import (
    create_rag_chain,
    load_and_split_documents,
    create_vector_store,
)

load_dotenv()

# Define the FastAPI app
app = FastAPI(
    title="On-Demand Document Q&A API",
    description="Send a document URL and questions to get answers using Hybrid Search.",
    version="3.0.0",
)

# Define request and response models
class ApiQueryRequest(BaseModel):
    document_url: str = Field(..., alias="documents", description="URL of the PDF document to process.")
    questions: List[str]

class Answer(BaseModel):
    question: str
    answer: str

class ApiResponse(BaseModel):
    answers: List[Answer]

# Create API endpoints
@app.post("/ask", response_model=ApiResponse)
async def ask_question(request: ApiQueryRequest):
    """
    Downloads a PDF from a URL, processes it, and answers questions based on its content.
    """
    temp_file_path = None
    try:
        # 1. Download the PDF from the URL
        print(f"Downloading document from: {request.document_url}")
        response = requests.get(request.document_url)
        response.raise_for_status()
        
        # 2. Save PDF content to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        # 3. Process the downloaded PDF
        chunks = load_and_split_documents(temp_file_path)
        vector_store = create_vector_store(chunks)

        # 4. Initialize the LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0
        )

        # 5. Create the RAG chain, passing in chunks for the keyword retriever
        rag_chain = create_rag_chain(chunks, vector_store, llm)
        
        # 6. Process all questions
        results = []
        for question in request.questions:
            print(f"Processing question: {question}")
            answer_text = rag_chain.invoke(question)
            results.append(Answer(question=question, answer=answer_text))
        
        return ApiResponse(answers=results)

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during processing.")
    finally:
        # 7. Manually clean up the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Cleaned up temporary file: {temp_file_path}")

@app.get("/", summary="Health Check")
async def health_check():
    """A simple health check endpoint to confirm the API is running."""
    return {"status": "ok"}