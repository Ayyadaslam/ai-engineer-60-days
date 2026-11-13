import os
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List

load_dotenv()

app = FastAPI()

client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# --- Models (Validation) ---
class ChatRequest(BaseModel):
    message: str

# Naya model extract request ke liye
class ExtractRequest(BaseModel):
    text: str

# Naya model AI ke JSON response ko validate karne ke liye
class ProfileResponse(BaseModel):
    name: str
    skills: List[str]
    experience_years: int
    current_role: str

# --- Endpoint 1: Streaming Chat (Day 2 Wala) ---
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


# --- Endpoint 2: Structured Output (Day 3 Wala) ---
@app.post("/extract", response_model=ProfileResponse)
async def extract_profile(request: ExtractRequest):
    print(f"Extracting from: {request.text}")
    
    # System prompt mein AI ko strictly JSON dene ka hukum lagaya hai
    system_prompt = """
    You are a data extraction AI. Your job is to extract information from the provided text.
    You MUST return ONLY a valid JSON object matching exactly this schema. Do not include any other text, markdown formatting, or explanations.
    
    Schema:
    {
        "name": "string",
        "skills": ["string", "string"],
        "experience_years": integer,
        "current_role": "string"
    }
    """
    
    try:
        response = await client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.text}
            ],
            # ✅ Yeh Groq/OpenAI ka feature hai jo strictly JSON output enforce karta hai
            response_format={"type": "json_object"} 
        )
        
        # AI ka text response
        json_string = response.choices[0].message.content
        print(f"AI Response: {json_string}")
        
        # String ko Python Dict (JSON) mein convert karein
        parsed_data = json.loads(json_string)
        
        # FastAPI khud isko ProfileResponse model se validate karega
        return parsed_data

    except Exception as e:
        return {"error": str(e)}