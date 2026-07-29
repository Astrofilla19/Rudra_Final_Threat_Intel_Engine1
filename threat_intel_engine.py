# threat_intel_engine.py
import json
import ollama
from typing import TypedDict, List, Dict, Any, Generator
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from langgraph.graph import StateGraph, END

# 1. IMPORT CONFIG FIRST (This injects the keys into os.environ)
from config import (
    DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    OLLAMA_MODEL_NAME,
)

# 2. IMPORT LANGFUSE SECOND (Now it can find the keys!)
from langfuse import observe

# ---------------------------------------------------------
# 1. STATE DEFINITION
# ---------------------------------------------------------
class AgentState(TypedDict):
    query: str
    documents: List[Dict[str, Any]]
    is_multi_video: bool
    context_str: str
    generation: str

def get_qdrant_client() -> QdrantClient:
    """Forces local disk storage since we are not running a Qdrant server."""
    return QdrantClient(path=str(DB_DIR))

# ---------------------------------------------------------
# 2. LANGGRAPH NODES
# ---------------------------------------------------------
@observe(name="Qdrant_Vector_Retrieval")
def retrieve_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    client = get_qdrant_client()

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query_vector = model.encode(query, show_progress_bar=False).tolist()

    try:
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=10,
        )
        search_results = response.points
    except Exception as e:
        print(f"❌ Qdrant Retrieval Error: {e}")
        return {"documents": [], "is_multi_video": False, "context_str": ""}

    if not search_results:
        return {"documents": [], "is_multi_video": False, "context_str": ""}

    video_groups: Dict[str, List[Dict[str, Any]]] = {}
    documents = []

    for point in search_results:
        if point.score < 0.25:
            continue
        payload = point.payload or {}
        vt = payload.get("video_title", "Unknown Target")
        doc_item = {
            "text": payload.get("text", ""),
            "time": payload.get("formatted_start", "00:00:00"),
            "video_title": vt,
            "score": point.score,
        }
        documents.append(doc_item)
        if vt not in video_groups:
            video_groups[vt] = []
        video_groups[vt].append(doc_item)

    unique_videos = list(video_groups.keys())
    is_multi_video = len(unique_videos) > 1

    # 🛑 NEW UNIFIED CONTEXT BUILDER
    context_blocks = []
    
    # Iterate through all videos and their respective chunks
    for vt, items in video_groups.items():
        for item in items:
            # Structurally bind the Video Title and Timestamp directly to every snippet
            context_blocks.append(
                f"[Video: {vt} | Timestamp: {item['time']}]\nSnippet: {item['text']}\n"
            )

    context_str = "\n---\n".join(context_blocks)
    
    return {
        "documents": documents,
        "is_multi_video": is_multi_video,
        "context_str": context_str,
    }


@observe(name="Ollama_LLM_Reasoning")
def generate_node(state: AgentState) -> Dict[str, Any]:
    
    context = state.get("context_str", "")
    query = state["query"]

    if not context:
        return {
            "generation": "❌ Analytical indicators failed to uncover matching entities in the vector store."
        }

    # 🛑 NEW STRICT SYSTEM PROMPT
    system_prompt = """
You are an advanced Cyber Threat Intelligence Reasoning Engine.

🚨 CRITICAL GROUNDING RULE: You must answer the user's query STRICTLY and ONLY using the facts provided in the Context Data below. 
- Do NOT use outside knowledge, prior training data, or external facts. 
- Do NOT hallucinate or guess. 
- If the Context Data does not contain the answer, you must output exactly: "Insufficient data in the ingested video feeds to answer this query."

You must structure your output report to ALWAYS lead with these three distinct fields at the very top:
1. THREAT CATEGORY: Choose exactly ONE word or short phrase that best captures the situational threat concern.
2. RISK LEVEL: Assign a tier out of: Critical, Very High, High, Medium, Low.
3. RISK SCORE: Assign a numeric score scale from 1 to 10.

4. ANALYTICAL BREAKDOWN:
Provide a concise, qualitative analytical breakdown of your findings based ONLY on the provided snippets. 
- 🔄 MULTI-SOURCE SYNTHESIS: If the context contains information from multiple different videos regarding the same topic, you MUST synthesize the information intelligently across them to provide a complete picture.
- 🛑 STRICT CITATION REQUIREMENT: Every single claim, fact, or analytical point you make MUST end with an inline structural citation identifying exactly where it came from.
- Format your citations exactly like this: (Source: [Video: <Exact Video Name> | Timestamp: <HH:MM:SS>]).
"""

    user_prompt = f"Context Data:\n{context}\n\nUser Intelligence Request: {query}"

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
        )
        generation = response["message"]["content"]
    except Exception as e:
        generation = f"❌ Ollama Connection Interrupted: {str(e)}"

    return {"generation": generation}

# ---------------------------------------------------------
# 3. LANGGRAPH WORKFLOW ASSEMBLY
# ---------------------------------------------------------
def build_generation_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

rag_agent = build_generation_graph()

# ---------------------------------------------------------
# 4. EXECUTION INTERFACES
# ---------------------------------------------------------
@observe(name="Run_CLI_Query_Pipeline")
def run_intelligence_query(query: str) -> str:
    print(f"\n🧠 Instantiating LangGraph Agent [{OLLAMA_MODEL_NAME}]...")
    
    initial_state: AgentState = {
        "query": query,
        "documents": [],
        "is_multi_video": False,
        "context_str": "",
        "generation": "",
    }
    
    final_state = rag_agent.invoke(initial_state)
    report = final_state.get("generation", "")
    
    print("\n" + "=" * 55 + "\n THREAT INTELLIGENCE ANALYSIS REPORT \n" + "=" * 55)
    print(report)
    print("\n" + "=" * 55 + "\n")
    return report


@observe(name="Stream_UI_Query_Pipeline")
def stream_intelligence_query(query: str, out_context: dict = None) -> Generator[str, None, None]:
    retrieval_output = retrieve_node(
        {"query": query, "documents": [], "is_multi_video": False, "context_str": "", "generation": ""}
    )
    
    context = retrieval_output.get("context_str", "")

    # [NEW]: This allows Streamlit to silently capture the context for evaluation!
    if out_context is not None:
        out_context["text"] = context

    if not context:
        yield "❌ Analytical indicators failed to uncover matching entities in Qdrant."
        return

    # 🛑 NEW STRICT SYSTEM PROMPT (Same as CLI)
    system_prompt = """
You are an advanced Cyber Threat Intelligence Reasoning Engine.

🚨 CRITICAL GROUNDING RULE: You must answer the user's query STRICTLY and ONLY using the facts provided in the Context Data below. 
- Do NOT use outside knowledge, prior training data, or external facts. 
- Do NOT hallucinate or guess. 
- If the Context Data does not contain the answer, you must output exactly: "Insufficient data in the ingested video feeds to answer this query."

You must structure your output report to ALWAYS lead with these three distinct fields at the very top:
1. THREAT CATEGORY: Choose exactly ONE word or short phrase that best captures the situational threat concern.
2. RISK LEVEL: Assign a tier out of: Critical, Very High, High, Medium, Low.
3. RISK SCORE: Assign a numeric score scale from 1 to 10.

4. ANALYTICAL BREAKDOWN:
Provide a concise, qualitative analytical breakdown of your findings based ONLY on the provided snippets. 
- 🔄 MULTI-SOURCE SYNTHESIS: If the context contains information from multiple different videos regarding the same topic, you MUST synthesize the information intelligently across them to provide a complete picture.
- 🛑 STRICT CITATION REQUIREMENT: Every single claim, fact, or analytical point you make MUST end with an inline structural citation identifying exactly where it came from.
- Format your citations exactly like this: (Source: [Video: <Exact Video Name> | Timestamp: <HH:MM:SS>]).
"""

    user_prompt = f"Context Data:\n{context}\n\nUser Intelligence Request: {query}"

    try:
        response_stream = ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
        )
        for chunk in response_stream:
            yield chunk["message"]["content"]
    except Exception as e:
        yield f"\n❌ Ollama Connection Interrupted: {str(e)}"