# utils/transcriber.py
import json
import torch
import whisper
import re
from pathlib import Path
from config import TRANSCRIPT_DIR, WHISPER_MODEL_NAME

def sanitize_filename(name: str) -> str:
    """Removes illegal characters from Windows file names."""
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()

def transcribe_audio(audio_path: str) -> str:
    print("🤖 Booting audio translation framework...")
    audio_path_obj = Path(audio_path)
    
    # 1. Sanitize the video title for Windows file saving
    safe_stem = sanitize_filename(audio_path_obj.stem)
    
    # 2. Force create the directory just in case it was deleted
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    
    output_path = TRANSCRIPT_DIR / f"{safe_stem}.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(WHISPER_MODEL_NAME, device=device)

    print(f"🗣️ Transcribing audio file using model architecture on choice: [{device}]...")
    result = model.transcribe(str(audio_path), verbose=False)

    segments = []
    for seg in result["segments"]:
        segments.append(
            {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=4)

    print("✅ Full textual translation compiled.")
    return str(output_path)