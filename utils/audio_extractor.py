# utils/audio_extractor.py
import subprocess
from pathlib import Path
from config import AUDIO_DIR


def extract_audio(video_path: str) -> str:
    print("🎵 Stripping acoustic tracks and re-sampling...")
    video_path_obj = Path(video_path)
    video_name = video_path_obj.stem
    audio_path = AUDIO_DIR / f"{video_name}.wav"

    command = [
        "ffmpeg",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(audio_path),
        "-y",
    ]

    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("✅ High-fidelity mono WAV track ready for speech processing.")
    return str(audio_path)