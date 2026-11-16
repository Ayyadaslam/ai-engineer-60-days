import os
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer, util

load_dotenv()

app = FastAPI()

# ==========================================
# 1. SETUP (Clients & Models)
# ==========================================
client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

print("RAG Models load ho rahe hain... (Thoda time lag sakta hai)")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Data Load aur Embeddings (Server start hote waqt ek dafa chalega)
try:
    with open("company_policy.txt", "r") as f:
        text_data = f.read()
    chunks = [line.strip() for line in text_data.split('\n') if line.strip()]
    chunk_embeddings = embed_model.encode(chunks)
    print("✅ Company Data successfully load aur embed ho gaya!")
except Exception as e:
    print(f"⚠️ Data load nahi hua: {e}")
    chunks = []
    chunk_embeddings = []

# ==========================================
# 2. SCHEMAS (Pydantic Models)
# ==========================================
class ChatRequest(BaseModel):
    message: str

class ExtractRequest(BaseModel):
    text: str

class ProfileResponse(BaseModel):
    name: str
    skills: List[str]
    experience_years: int
    current_role: str

class DocQuery(BaseModel):
    question: str

# ==========================================
# 3. ENDPOINTS
# ==========================================

# --- Day 2: Streaming Chat ---
async def stream_ai_response(user_message: str):
    try:
        response = await client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are a professional AI Engineer assistant. Always respond in English or Roman Urdu."},
                {"role": "user", "content": user_message}
            ],
            stream=True 
        )
        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"\n[Server Error: {str(e)}]"

@app.post("/chat")
async def chat_with_ai(request: ChatRequest): 
    return StreamingResponse(stream_ai_response(request.message), media_type="text/plain")


# --- Day 3: Structured Output ---
@app.post("/extract", response_model=ProfileResponse)
async def extract_profile(request: ExtractRequest):
    system_prompt = """
    You are a data extraction AI. Return ONLY a valid JSON object matching this schema:
    {"name": "string", "skills": ["string"], "experience_years": integer, "current_role": "string"}
    """
    try:
        response = await client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": request.text}],
            response_format={"type": "json_object"} 
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}


# --- Day 6: RAG API (Chat with Company Policy) ---
@app.post("/ask-docs")
async def ask_documents(request: DocQuery):
    print(f"User ka sawal: {request.question}")
    
    if not chunks:
        return {"error": "Data file not loaded properly."}

    # 1. User ke sawal ka vector banayein
    query_embedding = embed_model.encode(request.question)
    
    # 2. Vector Search (Best match dhoondein)
    similarities = util.cos_sim(query_embedding, chunk_embeddings)[0]
    best_match_idx = similarities.argmax()
    best_chunk = chunks[best_match_idx]
    
    # 3. LLM ko Context aur Query bhejein
    system_prompt = f"""
    You are a helpful HR assistant. Answer the user's question STRICTLY based on the context provided below. 
    If the answer is not in the context, say "Mujhe iski maloomat nahi hai."
    
    Context:
    {best_chunk}
    """
    
    try:
        response = await client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.question}
            ]
        )
        answer = response.choices[0].message.content
        
        return {
            "question": request.question,
            "retrieved_context": best_chunk, # Recruiters ko dikhane ke liye ke RAG kaam kar raha hai
            "answer": answer
        }
    except Exception as e:
        return {"error": str(e)}