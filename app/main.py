import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

from app.chunker import chunk_text
from app.embeddings import create_embeddings
from app.pdf_loader import load_pdf
from app.retriever import retrieve_chunks

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:63342",
        "http://127.0.0.1:63342",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


# -----------------------------
# Load resume and create vectors
# -----------------------------

pdf_path = Path(__file__).parent.parent / "data" / "resume.pdf"

resume_text = load_pdf(str(pdf_path))

chunks = chunk_text(resume_text)

embeddings = create_embeddings(chunks)


# -----------------------------
# Health check
# -----------------------------

@app.get("/")
def health_check():
    return {
        "status": "Portfolio AI Backend is running"
    }


# -----------------------------
# Chat request
# -----------------------------

class ChatRequest(BaseModel):
    query: str


@app.post("/chat")
def chat(request: ChatRequest):
    print(f'query: {request.query}')
    results = retrieve_chunks(
        query=request.query,
        chunks=chunks,
        embeddings=embeddings,
        top_k=1
    )

    context = results[0]["chunk"]

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Ravi Krishna Battala's personal portfolio assistant. "
                    "Answer questions using the provided resume context. "
                    "If the answer is not present in the context, say that "
                    "the information is not available in the provided resume."
                )
            },
            {
                "role": "user",
                "content": f"""
                    Resume context: {context}
                    Question: {request.query} + " in 2 lines"
                """
            }
        ]
    )

    return {
        "answer": response.choices[0].message.content
    }