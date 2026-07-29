# config.py
import os
from pathlib import Path
import torch

# ---------------------------------------------------------
# DIRECTORY PATHS & WORKSPACE SETUP
# ---------------------------------------------------------
# Base Directory of the Project
APP_ROOT = Path(__file__).resolve().parent

# Data Asset Directories
DATA_DIR = APP_ROOT / "data"
AUDIO_DIR = DATA_DIR / "audios"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
CHUNKS_DIR = DATA_DIR / "chunks"
DB_DIR = DATA_DIR / "vector_db"  # Fallback for local Qdrant disk storage

# Ensure all workspace directories exist safely
for folder in [AUDIO_DIR, TRANSCRIPT_DIR, CHUNKS_DIR, DB_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# HARDWARE STATE DETECTION
# ---------------------------------------------------------
if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"🚀 High-Performance Cluster Detected ({vram_gb:.1f}GB VRAM). Executing at full scale...")
else:
    print(f"⚠️ Local/CPU Environment Detected. Heavy models will run on CPU...")

# ---------------------------------------------------------
# MODEL CONSTANTS
# ---------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
WHISPER_MODEL_NAME = "large-v3-turbo"
OLLAMA_MODEL_NAME = "qwen2.5:32b"

# ---------------------------------------------------------
# VECTOR STORE CONFIGURATION (QDRANT)
# ---------------------------------------------------------
COLLECTION_NAME = "threat_intel_videos"

# ---------------------------------------------------------
# OBSERVABILITY CONFIGURATION (LANGFUSE)
# ---------------------------------------------------------
# MUST be injected directly into os.environ for the Langfuse SDK to detect them
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-436a2e8a-78bd-4c0c-8ca9-f367b265f7c6"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-2183771c-d208-4497-b77a-50589b6e0692"
os.environ["LANGFUSE_HOST"] = "https://jp.cloud.langfuse.com"