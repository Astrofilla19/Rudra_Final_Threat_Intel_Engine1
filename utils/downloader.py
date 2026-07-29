# utils/downloader.py
import yt_dlp
from config import AUDIO_DIR

def download_youtube_audio(url: str) -> str:
    print(f"📥 Fetching raw audio stream...")
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "outtmpl": str(AUDIO_DIR / "%(title)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")