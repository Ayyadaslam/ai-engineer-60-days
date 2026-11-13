import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

load_dotenv()

# 1. Embedding Model aur Groq Client Load Karein
print("Models load ho rahe hain...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# 2. Data File Read Karein aur Chunks Banayein
with open("company_policy.txt", "r") as f:
    text = f.read()

# Text ko chote paragraphs (chunks) mein todna
chunks = [line.strip() for line in text.split('\n') if line.strip()]

# Har chunk ka embedding banakar save karein
print("Data ko vectors mein convert kiya ja raha hai...")
chunk_embeddings = embed_model.encode(chunks)

def chat_with_data(user_query):
    print(f"\nUser: {user_query}")
    
    # 3. User ke sawal ka vector banayein
    query_embedding = embed_model.encode(user_query)
    
    # 4. Vector Search: Data mein se sabse milta-julta chunk dhoondein
    similarities = util.cos_sim(query_embedding, chunk_embeddings)[0]
    best_match_idx = similarities.argmax()
    best_chunk = chunks[best_match_idx]
    
    print(f"🔍 Database se yeh context mila: \"{best_chunk}\"")
    print("AI jawab soch raha hai...")

    # 5. LLM ko Context aur Query bhejein
    system_prompt = f"""
    You are a helpful HR assistant. Answer the user's question STRICTLY based on the context provided below. 
    If the answer is not in the context, say "Mujhe iski maloomat nahi hai."
    
    Context:
    {best_chunk}
    """
    
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
    )
    
    print(f"🤖 AI: {response.choices[0].message.content}\n")

# --- Test Karein ---
if __name__ == "__main__":
    chat_with_data("How many leaves do I get?") # English query
    chat_with_data("Laptop kon deta hai?")        # Roman Urdu query
    chat_with_data("Kya company car deti hai?")   # Aisi cheez jo data mein nahi hai