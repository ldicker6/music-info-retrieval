import os
import yt_dlp
from pydub import AudioSegment

def download_and_convert_youtube_audio(url, output_path="query_audio.wav"):
    temp_mp3 = "temp_youtube_audio.mp3"

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'temp_youtube_audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'ffmpeg_location': '/opt/homebrew/bin',  # or wherever your ffmpeg is installed
    }

    print("🎵 Downloading audio from YouTube...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    print("🎧 Converting to WAV...")
    audio = AudioSegment.from_mp3(temp_mp3)
    audio = audio.set_channels(1).set_frame_rate(22050)
    audio.export(output_path, format="wav")

    os.remove(temp_mp3)
    print(f"✅ Saved query audio as: {output_path}")
    return output_path
