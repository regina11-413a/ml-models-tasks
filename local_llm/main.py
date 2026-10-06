import requests

def ask(question: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:0.5b",
            "prompt": question,
            "stream": False
        }
    )
    return response.json()["response"]

if __name__ == "__main__":
    print("=" * 60)
    print("ЛОКАЛЬНАЯ LLM (Qwen2.5-0.5B через Ollama)")
    print("=" * 60)
    print("Введите 'exit' или 'выход' для выхода.\n")

    while True:
        question = input("Вопрос: ").strip()
        if question.lower() in ("exit", "quit", "выход"):
            break
        if not question:
            continue
        print(f"\nОтвет: {ask(question)}\n")
        print("-" * 60)