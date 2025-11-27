import sys
from youtube_audio_utils import download_and_convert_youtube_audio
from query_interface import run_query_agent  # wrap main logic as function

def main():
    if len(sys.argv) < 2:
        print("Usage: python query_youtube.py <YouTube URL>")
        return
    url = sys.argv[1]
    wav_path = download_and_convert_youtube_audio(url, output_path="query_audio.wav")
    run_query_agent(wav_path)
if __name__ == "__main__":
    main()
