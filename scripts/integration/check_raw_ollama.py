import requests

from src.config import settings


def test_raw_inference(prompt):
    url = f"{settings.OLLAMA_BASE_URL}/api/generate"
    payload = {"model": "llama3.2:3b", "prompt": prompt, "stream": False, "options": {"temperature": 0.7}}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        return f"Error: {e}"


def main():
    print("🧪 Testing Raw Llama 3.2 3B Inference")

    prompts = [
        "Translate 'Hello, how are you?' to Hindi.",
        "Translate 'Hello, how are you?' to Kannada.",
        "Translate 'Hello, how are you?' to Telugu.",
        "Translate 'Hello, how are you?' to Tamil.",
        "Write a short sentence in Kannada.",
    ]

    for p in prompts:
        print(f"\n📥 Prompt: {p}")
        resp = test_raw_inference(p)
        print(f"🤖 Response: {resp.strip()}")


if __name__ == "__main__":
    main()
