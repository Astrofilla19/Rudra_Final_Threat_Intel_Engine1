# eval_generation.py
import json
import ollama

# [FIXED]: Updated to match your installed Langfuse version
from langfuse import observe 
from config import OLLAMA_MODEL_NAME

@observe(name="LLM_as_a_Judge_Evaluation")
def evaluate_generation(
    query: str, context: str, generated_answer: str, ground_truth: str
) -> dict:
    """
    Uses an LLM-as-a-Judge to grade Faithfulness, Answer Relevancy, and Answer Correctness.
    """

    eval_system_prompt = """
You are an impartial Data Engineering Evaluator grading a RAG-based Threat Intelligence system.
You must output strictly valid JSON with no markdown tags or extra text.

Evaluate the Generated Answer on three metrics, scoring each from 1 to 5:
1. FAITHFULNESS: Is the Generated Answer entirely supported by the Provided Context? (1 = Contains hallucinations, 5 = Strictly grounded in context).
2. ANSWER RELEVANCY: Does the Generated Answer directly and concisely address the User Query? (1 = Off-topic/rambling, 5 = Highly precise and direct).
3. ANSWER CORRECTNESS: Does the Generated Answer semantically match the facts in the Ground Truth Answer? (1 = Completely wrong, 5 = Factually identical).

JSON Schema Requirement:
{"Faithfulness": <int>, "Answer_Relevancy": <int>, "Answer_Correctness": <int>, "Reasoning": "<string brief explanation>"}
"""

    user_prompt = f"""
[User Query]: {query}
[Provided Context]: {context}
[Ground Truth Answer]: {ground_truth}
[Generated Answer]: {generated_answer}
"""

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[
                {"role": "system", "content": eval_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            format="json",
        )

        raw_content = response["message"]["content"].strip()

        # [UPDATE]: Robust Markdown Fence Stripping to prevent JSON parsing crashes
        if raw_content.startswith("```"):
            raw_content = raw_content.strip("` \n")
            if raw_content.startswith("json"):
                raw_content = raw_content[4:].strip()

        evaluation_result = json.loads(raw_content)
        return evaluation_result

    except Exception as e:
        print(f"❌ LLM Evaluation Failed: {e}")
        return {
            "Faithfulness": 0,
            "Answer_Relevancy": 0,
            "Answer_Correctness": 0,
            "Reasoning": f"Error during generation evaluation: {str(e)}",
        }


if __name__ == "__main__":
    test_query = "What percentage of petrol facilities were destroyed?"
    test_context = "...knocked out roughly 25% of petrol production..."
    test_generated = "25% of petrol facilities were destroyed."
    test_truth = "Roughly 25% of petrol production facilities were destroyed."

    scores = evaluate_generation(test_query, test_context, test_generated, test_truth)
    print(json.dumps(scores, indent=4))