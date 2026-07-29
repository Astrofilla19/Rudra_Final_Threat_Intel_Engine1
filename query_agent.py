# query_agent.py
from threat_intel_engine import run_intelligence_query

def start_cli_agent():
    print("=" * 60)
    print(" 🛡️ THREAT INTELLIGENCE AGENT - CLI QUERY INTERFACE")
    print("=" * 60)

    while True:
        print("-" * 55)
        query = input("Ask a Threat Intelligence Query (or type 'exit' / 'quit'): ").strip()

        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting query agent runtime.")
            break
        if not query:
            continue

        try:
            run_intelligence_query(query)
        except Exception as e:
            print(f"❌ Query execution failed: {e}")

if __name__ == "__main__":
    start_cli_agent()