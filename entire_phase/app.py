import argparse
import sys
from pathlib import Path

# Add the parent directory to Python path to import from Qna chatbot
sys.path.append(str(Path(__file__).resolve().parent.parent / "Qna chatbot"))

from core.single_product import answer

def main():
    parser = argparse.ArgumentParser(description="Single-product chatbot CLI")
    parser.add_argument("--question", "-q", type=str)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--provider", type=str, choices=["anthropic","openai"], default=None)
    args = parser.parse_args()

    q = args.question or input("Enter your question: ").strip()
    if not q:
        print("No question provided.")
        return

    print(answer(q, top_k=args.top_k, limit=args.limit, provider=args.provider))

if __name__ == "__main__":
    main()
