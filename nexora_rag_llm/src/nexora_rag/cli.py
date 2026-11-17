import argparse, json
from .ingestion import ingest_policies
from .pipeline import NexoraAssistant

def main():
    parser = argparse.ArgumentParser(description="Nexora Bank RAG + LLM demo")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="Extract, chunk, embed and index policy PDFs")
    ingest.add_argument("--reset", action="store_true")
    ask = sub.add_parser("ask", help="Retrieve policies and draft a human-review message")
    ask.add_argument("--question", required=True)
    ask.add_argument("--category", default="moderate",
                     choices=["consistent", "moderate", "declining_activity", "dormant", "billing_reminder"])
    ask.add_argument("--customer-facts", default="{}", help="JSON object of verified facts")
    ask.add_argument("--checks", default="{}", help="JSON object of deterministic check results")
    ask.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()
    if args.command == "ingest":
        print(json.dumps(ingest_policies(reset=args.reset), indent=2))
    elif args.command == "ask":
        result = NexoraAssistant().answer(
            args.question, args.category,
            json.loads(args.customer_facts), json.loads(args.checks), args.top_k
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
