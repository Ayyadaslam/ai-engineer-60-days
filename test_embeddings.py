from sentence_transformers import SentenceTransformer, util

# 1. Model Load Karein
print("Model load ho raha hai... (Thoda time lag sakta hai)")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Sentences ki list (Pure English)
sentences = [
    "I am a web developer skilled in React and Node.js.",    # Sentence 1
    "MERN stack is used for full-stack web development.",    # Sentence 2
    "I love eating chicken biryani on weekends."             # Sentence 3
]

# 3. In sentences ko Embeddings (Vectors) mein convert karein
print("Embeddings generate ho rahi hain...")
embeddings = model.encode(sentences)

# 4. Similarity (Milta-julta hona) check karein
print("\n--- Similarity Check ---")
sim_1_2 = util.cos_sim(embeddings[0], embeddings[1])
sim_1_3 = util.cos_sim(embeddings[0], embeddings[2])

print(f"Sentence 1 (Developer) vs Sentence 2 (Web Dev) : {sim_1_2.item():.4f}  --> High")
print(f"Sentence 1 (Developer) vs Sentence 3 (Biryani) : {sim_1_3.item():.4f}  --> Low")