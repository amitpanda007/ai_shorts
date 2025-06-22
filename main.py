import os
import yt_dlp
import whisper
import moviepy.editor as mp
from moviepy.video.tools.subtitles import SubtitlesClip
import moviepy.config as cf
from openai import OpenAI
from dotenv import load_dotenv
import json
import ollama
import requests
import shutil

# This line is specific to your system and can be kept.
# It's only necessary if moviepy can't find ImageMagick automatically for TextClip rendering.
cf.IMAGEMAGICK_BINARY = r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"

# --- Configuration ---
OUTPUT_DIR = "output"
WHISPER_MODEL_SIZE = "base"
# NOTE: Make sure you have pulled this specific model, e.g., `ollama pull llama3` or `ollama pull mistral`
# Using llama3 is recommended as it's very good at following JSON instructions.
OLLAMA_MODEL = "llama3.2"

# --- Setup ---
load_dotenv()
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Determine which AI service to use
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
OLLAMA_ENABLED = False

if USE_OPENAI:
    print("OpenAI API key found. Using OpenAI for highlight detection.")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:
    try:
        requests.get("http://localhost:11434", timeout=2)
        OLLAMA_ENABLED = True
        print(f"OpenAI key not found. Ollama server detected. Using Ollama model '{OLLAMA_MODEL}'.")
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        print("OpenAI key not found and Ollama server is not running. Falling back to simple highlight detection.")

# --- Unchanged Functions (download_video, transcribe_audio) ---
def download_video(url: str) -> str:
    """Downloads a YouTube video in the highest resolution and returns the file path."""
    print(f"Downloading video from URL: {url}")
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg and ensure it is in your system's PATH."
        )
    output_template = os.path.join(OUTPUT_DIR, '%(title)s.%(ext)s')
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
        'outtmpl': output_template,
        'verbose': True,
        'nocheckcertificate': True,
        'hls_prefer_ffmpeg': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.cache.remove()
        except Exception:
            pass # Ignore if cache doesn't exist
        info = ydl.extract_info(url, download=True)
        info['ext'] = 'mp4'
        filename = ydl.prepare_filename(info)
        if not os.path.exists(filename) or os.path.getsize(filename) == 0:
            raise yt_dlp.utils.DownloadError("File was not created or is empty after download attempt.")
        print(f"Video downloaded to: {filename}")
        return filename

def transcribe_audio(video_path: str) -> dict:
    """Transcribes the audio of a video file and returns word-level timestamps."""
    print(f"Loading Whisper model '{WHISPER_MODEL_SIZE}'...")
    model = whisper.load_model(WHISPER_MODEL_SIZE)
    print("Transcribing audio... (This may take a while)")
    result = model.transcribe(video_path, word_timestamps=True)
    transcription_path = os.path.splitext(video_path)[0] + ".json"
    with open(transcription_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Transcription complete. Saved to {transcription_path}")
    return result

# --- 3. Highlight Detection (With Improved Prompt) ---

def find_highlights_with_llm(transcription: dict, num_clips: int, min_duration: int, max_duration: int, use_openai: bool) -> list:
    """
    Uses an LLM (Ollama or OpenAI) to find complete, engaging segments.
    """
    ai_service = "OpenAI" if use_openai else f"Ollama ({OLLAMA_MODEL})"
    print(f"Finding highlights with {ai_service}...")

    # We now provide the full text with timestamps for context.
    formatted_transcript = ""
    for segment in transcription['segments']:
        start = int(segment['start'])
        text = segment['text']
        formatted_transcript += f"[{start}s] {text.strip()}\n"

    ### MODIFICATION 2: A much better, more specific prompt ###
    prompt = f"""
    You are an expert viral video editor. Your task is to identify complete, compelling moments in a video transcript.
    From the following transcript, please select the {num_clips} most engaging or highlight-worthy segments.

    **CRITICAL RULES:**
    1. Each clip's duration MUST be between {min_duration} and {max_duration} seconds.
    2. The "end" time for each clip MUST align with the end of a sentence or a complete thought. Do NOT cut off in the middle of a sentence.
    3. Find segments that are engaging, informative, or have a strong emotional peak.

    Your response MUST be a valid JSON array of objects. Each object must have a "start" and "end" time in seconds.
    Do NOT include any other text, explanations, or markdown code fences like ```json. Your output must start with `[` and end with `]`.

    Example format:
    [
        {{"start": 122, "end": 165}},
        {{"start": 345, "end": 398}}
    ]

    Transcript with timestamps:
    ---
    {formatted_transcript}
    ---
    """

    try:
        if use_openai:
            response = client.chat.completions.create(
                model="gpt-4o-mini", # Using a more advanced model can yield better results
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.5,
            )
            content = response.choices[0].message.content
        else: # Use Ollama
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                format='json'
            )
            content = response['message']['content']

        # The response should be a JSON object containing a key, e.g., {"clips": [...]}. We need to find the array.
        data = json.loads(content)
        highlights = []

        if isinstance(data, list):
            # Case 1: The LLM returned a direct list of objects.
            highlights = data
        elif isinstance(data, dict):
            # Case 2: The LLM returned a dictionary.
            # Check if it contains a list (e.g., {"clips": [...]})
            list_from_dict = next((v for v in data.values() if isinstance(v, list)), None)
            if list_from_dict is not None:
                highlights = list_from_dict
            # Check if it's a single clip object (e.g., {"start": ...})
            elif 'start' in data and 'end' in data:
                highlights = [data]  # <--- Wrap the single dictionary in a list

        if not highlights:
            raise ValueError("Could not parse a valid list of clips from the LLM response.")

        print(f"{ai_service} identified highlights: {highlights}")
        return highlights
    except Exception as e:
        print(f"Error communicating with {ai_service} or parsing its response: {e}")
        # We are removing the simple fallback to make the AI's role more critical.
        # If you want a fallback, you can call find_highlights_simple here.
        return []


def create_clip_with_subtitles(video_path: str, transcription: dict, start_time: int, end_time: int, clip_num: int):
    """Creates a video clip and burns subtitles onto it."""
    print(f"Creating clip {clip_num} from {start_time}s to {end_time}s...")

    video = mp.VideoFileClip(video_path)
    clip = video.subclip(start_time, end_time)

    def subtitle_generator(txt):
        words_in_clip = []
        for segment in transcription['segments']:
            for word_info in segment['words']:
                word_start, word_end = word_info['start'], word_info['end']
                if word_start >= start_time and word_end <= end_time:
                    relative_start = word_start - start_time
                    relative_end = word_end - start_time
                    words_in_clip.append(((relative_start, relative_end), word_info['word']))
        return words_in_clip

    subtitles = SubtitlesClip(subtitle_generator(transcription),
                              lambda txt: mp.TextClip(txt, font='Arial-Bold', fontsize=60, color='white',
                                                      stroke_color='black', stroke_width=2, method='caption', size=(clip.w * 0.8, None)))

    ### MODIFICATION 1: Change subtitle position ###
    # We move the subtitles from 'bottom' to 75% of the screen height, which is a common position for shorts.
    final_clip = mp.CompositeVideoClip([clip, subtitles.set_position(('center', 0.85), relative=True)])


    (w, h) = final_clip.size
    target_ratio = 9.0 / 16.0
    current_ratio = w / h
    if current_ratio > target_ratio:
        new_width = int(h * target_ratio)
        final_clip = final_clip.crop(x_center=w/2, width=new_width)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = os.path.join(OUTPUT_DIR, f"{base_name}_clip_{clip_num}.mp4")

    final_clip.write_videofile(
        output_filename, codec="libx264", audio_codec="aac", temp_audiofile='temp-audio.m4a', remove_temp=True,
        preset="medium", bitrate="5000k", threads=4, fps=video.fps
    )

    print(f"Successfully created clip: {output_filename}")
    video.close()


def main_workflow(url: str, num_clips: int, min_duration: int, max_duration: int):
    """The main function to orchestrate the entire process."""
    try:
        video_path = download_video(url)
        transcription = transcribe_audio(video_path)

        # Unified logic to select the highlight function
        if USE_OPENAI or OLLAMA_ENABLED:
             highlights = find_highlights_with_llm(transcription, num_clips, min_duration, max_duration, use_openai=USE_OPENAI)
        else:
            print("No AI service available (OpenAI or Ollama). Cannot find highlights.")
            highlights = []

        if not highlights:
            print("No highlights were found. Exiting.")
            return

        video_duration = mp.VideoFileClip(video_path).duration
        for i, h in enumerate(highlights):
            start = h['start']
            end = min(h['end'], video_duration)
            if start >= end:
                print(f"Skipping invalid clip {i+1} with start >= end time.")
                continue

            create_clip_with_subtitles(video_path, transcription, start, end, i + 1)

        print("\nAll clips generated successfully!")

    except Exception as e:
        import traceback
        print(f"\nAn error occurred in the main workflow: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    YOUTUBE_URL = "https://www.youtube.com/watch?v=4H2wCY2Ulp4"
    NUM_CLIPS = 2
    ### MODIFICATION 2 (continued): Using a duration range ###
    MIN_CLIP_DURATION = 30 # in seconds
    MAX_CLIP_DURATION = 60 # in seconds

    main_workflow(YOUTUBE_URL, NUM_CLIPS, MIN_CLIP_DURATION, MAX_CLIP_DURATION)