import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ✅ Change 1: base_url add karein (OpenAI SDK ko Gemini se connect karne ke liye)
# ✅ Change 2: API key Gemini wali use karein
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def ask_ai(user_message):
    print(f"User: {user_message}")
    print("AI soch raha hai...")

    response = client.chat.completions.create(
        model="gemini-3.8-flash",  # ✅ Change 3: Gemini ka free model use karein
        messages=[
            {"role": "system", "content": "You are a professional AI Engineer assistant."},
            {"role": "user", "content": user_message}
        ]
    )

    ai_reply = response.choices[0].message.content
    print(f"AI: {ai_reply}\n")

if __name__ == "__main__":
    ask_ai("Mujhe AI Engineer banne ke liye sabse pehla step kya lena chahiye?")