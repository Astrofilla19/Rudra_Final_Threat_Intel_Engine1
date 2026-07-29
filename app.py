# app.py
import streamlit as st
import traceback
from pathlib import Path

# Import your updated pipeline modules
from utils.downloader import download_youtube_audio
from utils.transcriber import transcribe_audio
from utils.semantic_chunker import generate_semantic_chunks
from utils.vector_store import ingest_to_vector_store
from threat_intel_engine import stream_intelligence_query
from run_eval_pipeline import run_automated_evaluation
from eval_generation import evaluate_generation  # [NEW] Importing your evaluator

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SURAKSHA Threat Intel", 
    page_icon="🛡️", 
    layout="wide"
)

st.title("🛡️ SURAKSHA")
st.markdown("**S**ystem for **U**nified **R**etrieval, **A**ssessment, & **K**nowledge **S**ynthesis in **H**azard **A**nalysis")

# Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR: DATA INGESTION & EVALUATION ---
with st.sidebar:
    st.header("📥 Ingest Target Feeds")
    st.markdown("Paste YouTube URLs below (one per line):")
    
    urls_input = st.text_area("Target URLs", height=150, placeholder="https://youtube.com/...")
    
    if st.button("🚀 Process Targets", use_container_width=True):
        urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
        
        if not urls:
            st.warning("Please enter at least one valid URL.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for index, url in enumerate(urls, 1):
                status_text.text(f"Processing {index}/{len(urls)}: {url}")
                try:
                    with st.spinner(f"Downloading & Transcribing Video {index}..."):
                        audio_file = download_youtube_audio(url)
                        video_title = Path(audio_file).stem
                        
                        transcript_json = transcribe_audio(audio_file)
                        chunks_json = generate_semantic_chunks(transcript_json)
                        
                        ingest_to_vector_store(chunks_json, video_title=video_title)
                        st.success(f"✅ {video_title} ingested successfully.")
                except Exception as e:
                    st.error(f"Failed to process {url}")
                    st.error(traceback.format_exc())
                
                progress_bar.progress(index / len(urls))
            
            status_text.text("Batch Processing Complete!")
            st.balloons()
            
    st.markdown("---")
    
    st.header("📊 Batch System Evaluation")
    st.markdown("Run the offline test suite against the Golden Dataset.")
    
    if st.button("⚙️ Run RAG Evaluation Suite", use_container_width=True):
        with st.spinner("Running Automated Evaluation..."):
            try:
                metrics = run_automated_evaluation()
                if metrics:
                    st.success("✅ Evaluation Complete!")
                    st.metric(label="Mean Hit Rate (Retrieval)", value=f"{metrics['Hit_Rate']:.3f}")
                    st.metric(label="Mean NDCG (Ranking)", value=f"{metrics['NDCG']:.3f}")
                    st.metric(label="Mean Faithfulness (No Hallucinations)", value=f"{metrics['Faithfulness']:.2f} / 5.0")
                    st.metric(label="Answer Relevancy", value=f"{metrics['Relevancy']:.2f} / 5.0")
                    st.metric(label="Answer Correctness", value=f"{metrics['Correctness']:.2f} / 5.0")
                else:
                    st.error("❌ Evaluation failed. Ensure golden_dataset.json exists.")
            except Exception as e:
                st.error(f"Evaluation Error: {str(e)}")

# --- MAIN WINDOW: CHAT INTERFACE ---
st.subheader("Intelligence Query Interface")

# Display historical chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user queries
if prompt := st.chat_input("Ask a Threat Intelligence Query..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Create an empty dictionary to catch the retrieved Qdrant context
        captured_context = {}
        stream = stream_intelligence_query(prompt, out_context=captured_context)
        response = st.write_stream(stream)
        
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Save the latest interaction to session state so we can evaluate it
    st.session_state.latest_eval_data = {
        "query": prompt,
        "context": captured_context.get("text", ""),
        "response": response
    }

# --- NEW: REAL-TIME ONLINE EVALUATION PANEL ---
if "latest_eval_data" in st.session_state and st.session_state.latest_eval_data["context"]:
    st.markdown("---")
    with st.expander("⚖️ Real-Time Response Evaluation (LLM-as-a-Judge)", expanded=True):
        st.markdown("**Test the accuracy of the last generated response against a Ground Truth.**")
        ground_truth_input = st.text_area("Enter the Expected / Real Answer here:", placeholder="The state of emergency was declared because...")
        
        if st.button("Score Response"):
            if not ground_truth_input.strip():
                st.warning("⚠️ Please provide a ground truth answer to evaluate against.")
            else:
                with st.spinner("🧠 Qwen 2.5 is currently grading the response..."):
                    eval_data = st.session_state.latest_eval_data
                    scores = evaluate_generation(
                        query=eval_data["query"],
                        context=eval_data["context"],
                        generated_answer=eval_data["response"],
                        ground_truth=ground_truth_input
                    )
                    
                    # Display the AI Judge's scores as clean dashboard metrics
                    cols = st.columns(3)
                    cols[0].metric("Faithfulness", f"{scores.get('Faithfulness', 0)} / 5")
                    cols[1].metric("Relevancy", f"{scores.get('Answer_Relevancy', 0)} / 5")
                    cols[2].metric("Correctness", f"{scores.get('Answer_Correctness', 0)} / 5")
                    
                    st.info(f"**AI Judge Reasoning:** {scores.get('Reasoning', 'No reasoning provided.')}")
                    st.success("✅ Evaluation successfully traced and logged directly to your Langfuse dashboard!")