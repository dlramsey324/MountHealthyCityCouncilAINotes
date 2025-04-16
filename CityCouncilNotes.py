import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import yt_dlp
from openai import OpenAI

# Initialize the OpenAI client with your API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def download_audio_from_youtube(url, output_path="audio"):
    """
    Downloads the audio from the provided YouTube URL and converts it to MP3.
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'noplaylist': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print(f"Audio downloaded and saved as {output_path}")


def transcribe_audio(audio_path):
    """
    Sends the MP3 file to OpenAI’s transcription API (Whisper) and returns the transcription.
    """
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model='whisper-1',
            language='en',   # specify english (prevents issue with some welsh embedded in
                             #    "waiting for meeting to start" programming
            file=audio_file
        )
    return transcription


def compress_audio(input_path, output_path, bitrate="64k"):
    import subprocess
    command = ["ffmpeg", "-i", input_path, "-b:a", bitrate, output_path, "-y"]
    subprocess.run(command, check=True)
    print(f"Compressed audio saved as {output_path}")


def split_audio(input_path, segment_duration=1800, output_prefix="segment_"):
    """
    Splits the input audio file into segments of given duration (in seconds).
    Each segment is saved with the given prefix and an incremental index.
    """
    import subprocess
    command = [
        "ffmpeg", "-i", input_path,
        "-f", "segment", "-segment_time", str(segment_duration),
        "-c", "copy", output_prefix + "%03d.mp3", "-y"
    ]
    subprocess.run(command, check=True)
    print(f"Audio split into segments with prefix {output_prefix}")


def generate_summary(transcribed_text):
    """
    Uses OpenAI Chat Completions API (model: 4o-mini) to generate a summary of the meeting notes.
    The summary emphasizes:
      - Roads near Madison Ave and in the southwest portion of the city
      - Additional roads: Park, Forest, Stevens, Maple, Werner, Harrison
      - Updates on city playgrounds, parks, or pools
      - Any other important details residents need to know
    """
    system_prompt = (
        "You are a clear, efficient summarizer of local government meeting notes. Your job is to produce a concise, structured summary that helps a local couple stay up to date on everything they care about from a city council meeting—without needing to listen to the full recording.\n\n"
        "Prioritize updates related to:\n"
        "- Road work and traffic changes near Madison Avenue and the southwest side of the city, especially Park, Forest, Stevens, Maple, Werner, and Harrison.\n"
        "- Parks, playgrounds, and pools—any maintenance, funding, improvements, or events.\n\n"
        "After that, include other decisions, programs, events, and changes that a resident would reasonably want to know. Make sure nothing important is left out."
    )

    user_prompt = (
        "Please summarize the following city council meeting transcript. The goal is to help my wife and me quickly stay informed about what matters most without listening to the full meeting.\n\n"
        "Focus on:\n"
        "- Road updates, especially around Madison Avenue and southwest city streets: Park, Forest, Stevens, Maple, Werner, and Harrison.\n"
        "- City parks, playgrounds, and pools.\n"
        "- All other key decisions or announcements that are relevant to staying informed as local residents.\n\n"
        f"Transcript:\n\n{transcribed_text}"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    summary = response.choices[0].message.content
    return summary


def main():
    # Set your YouTube video URL (change as needed)
    youtube_url = "https://www.youtube.com/watch?v=Dgsxh3dO9lE&list=PLEuWRfzQpk8WtZbkDKTKB2sC8XC-UTjox&index=1"
    audio_base = "audio"   # Base name without extension
    audio_file = audio_base + ".mp3"   # Final audio file name after postprocessing

    # Step 1: Download and convert audio
    print("Downloading audio from YouTube...")
    download_audio_from_youtube(youtube_url, audio_base)

    # Step 2: Split the audio into 30 minute segments
    print("Splitting audio into 30 minute segments...")
    split_audio(audio_file, segment_duration=1800, output_prefix="segment_")

    # Step 3: Transcribe each audio segment and combine the transcripts
    import glob
    segment_files = sorted(glob.glob("segment_*.mp3"))
    full_transcript = ""
    for segment in segment_files:
        print(f"Compressing {segment} to reduce file size...")
        compressed_segment = "compressed_" + segment
        compress_audio(segment, compressed_segment, bitrate="64k")
        print(f"Transcribing {compressed_segment}...")
        transcription = transcribe_audio(compressed_segment)
        full_transcript += transcription.text + "\n"
    print("Full transcript generated:")
    print(full_transcript)

    # Step 4: Generate a summary from the full transcript
    print("Generating summary using OpenAI Chat Completions API...")
    generated_summary = generate_summary(full_transcript)
    print("Generated summary from the model:")
    print(generated_summary)


if __name__ == "__main__":
    main()
