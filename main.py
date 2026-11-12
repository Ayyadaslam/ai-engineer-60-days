import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

# ✅ Groq ka OpenAI-compatible client
client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

class ChatRequest(BaseModel):
    message: str

async def stream_ai_response(user_message: str):
    try:
        response = await client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                # ✅ Yahan change kiya hai: AI ko strictly Roman Urdu/English bolne ka kaha hai
                {"role": "system", "content": "You are a professional AI Engineer assistant. Always respond in English or Roman Urdu. Never use Hindi, Devanagari, or any other regional script."},
                {"role": "user", "content": user_message}
            ],
            stream=True 
        )
        
        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        print(f"❌ Server Error: {e}")
        yield f"\n[Server Error: {str(e)}]"

@app.post("/chat")
async def chat_with_ai(request: ChatRequest): 
    print(f"User ne bheja: {request.message}")
    return StreamingResponse(stream_ai_response(request.message), media_type="text/plain")