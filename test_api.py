import requests

url = "http://127.0.0.1:8000/chat"
payload = {"message": "give me the  road map of ai software engineer"}
headers = {"Content-Type": "application/json"}

print("AI ka jawab aa raha hai:\n")

try:
    with requests.post(url, json=payload, headers=headers, stream=True) as response:
        if response.status_code != 200:
            print(f"❌ API Error: Status {response.status_code} - {response.text}")
        else:
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    print(chunk.decode('utf-8'), end="", flush=True)
                
except Exception as e:
    print(f"\n\n❌ Python Error: {e}")

print("\n\n--- Stream Ended ---")