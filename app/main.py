import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

from app.chunker import chunk_text
from app.embeddings import create_embeddings
from app.retriever import retrieve_chunks
from app.document_loader import load_documents

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ravikrishnabattala.netlify.app",
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

data_path = Path(__file__).parent.parent / "data"

knowledge_text = load_documents(str(data_path))

chunks = chunk_text(knowledge_text)

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
        top_k=2
    )

    context = "\n\n".join(
        result["chunk"]
        for result in results
    )

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Ravi Krishna Battala's personal "
                    "portfolio assistant.\n\n"
                    "Answer questions using the provided portfolio "
                    "knowledge context.\n"
                    "If the answer is not available in the provided "
                    "context, say that the information is not available "
                    "in the provided portfolio knowledge."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Knowledge context: {context}\n"
                    f"Question: {request.query}\n"
                    "Answer in 2 concise lines."
                )
            }
        ]
    )

    return {
        "answer": response.choices[0].message.content
    }