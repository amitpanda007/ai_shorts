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
from google import genai
from google.genai import types
import re
import glob

# This line is specific to your system and can be kept.
cf.IMAGEMAGICK_BINARY = r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"

# --- Configuration ---
OUTPUT_DIR = "output"
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
WHISPER_MODEL_SIZE = "base"
OLLAMA_MODEL = "llama3"
CHUNK_DURATION = 300  # 5 minutes

# --- Setup ---
load_dotenv()
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Determine which AI service to use and configure it
AI_SERVICE = None
if os.getenv("OPENAI_API_KEY"):
    AI_SERVICE = "openai"
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("AI Service: OpenAI")
elif os.getenv("GEMINI_API_KEY"):
    AI_SERVICE = "gemini"
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY")
    )
    print("AI Service: Gemini")
else:
    try:
        requests.get("http://localhost:11434", timeout=2)
        AI_SERVICE = "ollama"
        print(f"AI Service: Ollama ({OLLAMA_MODEL})")
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        print("AI Service: None. Please configure OpenAI, Gemini, or a running Ollama instance.")


class ClipCounter:
    def __init__(self): self.count = 0

    def get(self): self.count += 1; return self.count


def download_video(url: str, download_path: str) -> str:
    """Downloads a YouTube video to a specific path."""
    print(f"Downloading video from URL: {url} to {download_path}")
    if not shutil.which("ffmpeg"): raise RuntimeError("ffmpeg not found.")

    # We provide the template without the extension, yt-dlp will add it.
    output_template = os.path.splitext(download_path)[0]

    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': output_template,
        'merge_output_format': 'mp4',
        'quiet': True,
        'nocheckcertificate': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # The final path will have the .mp4 extension added by yt-dlp
    final_path = output_template + ".mp4"
    if not os.path.exists(final_path) or os.path.getsize(final_path) == 0:
        raise yt_dlp.utils.DownloadError("Downloaded file is missing or empty.")
    return final_path


def transcribe_audio(video_path: str) -> dict:
    # This function remains unchanged.
    print(f"Loading Whisper model '{WHISPER_MODEL_SIZE}'...")
    model = whisper.load_model(WHISPER_MODEL_SIZE)
    print(f"Transcribing audio for {video_path}...")
    result = model.transcribe(video_path, word_timestamps=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    transcription_path = os.path.join(OUTPUT_DIR, f"{base_name}_transcription.json")
    with open(transcription_path, 'w', encoding='utf-8') as f: json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Transcription complete. Saved to {transcription_path}")
    return result


def _parse_llm_response(content: str) -> list:
    """
    Parses and sanitizes the LLM's JSON response, removing markdown fences.
    """
    print(f"LLM Raw Response: '{content}'")  # Use quotes to see leading/trailing characters

    # Sanitize the response: remove markdown fences and trim whitespace
    sanitized_content = content.strip()
    if sanitized_content.startswith("```json"):
        sanitized_content = sanitized_content[7:]
    elif sanitized_content.startswith("```"):
        sanitized_content = sanitized_content[3:]

    if sanitized_content.endswith("```"):
        sanitized_content = sanitized_content[:-3]

    sanitized_content = sanitized_content.strip()

    print(f"Sanitized Content for JSON parsing: '{sanitized_content}'")

    data = json.loads(sanitized_content)
    highlights = []
    if isinstance(data, dict) and 'clips' in data and isinstance(data['clips'], list):
        highlights = data['clips']
    elif isinstance(data, list):
        highlights = data
    elif isinstance(data, dict) and 'start' in data and 'end' in data:
        highlights = [data]
    if not highlights or not all('start' in d and 'end' in d for d in highlights):
        raise ValueError("LLM did not return the expected format of clip objects.")
    return highlights


### UNIFIED AND REFACTORED LLM FUNCTION ###
def find_highlights_with_llm(transcription: dict, num_clips: int, min_duration: int, max_duration: int,
                             ai_service: str) -> list:
    """A single, unified function to interact with any configured LLM."""
    print(f"Finding highlights with {ai_service.capitalize()}...")
    formatted_transcript = ""
    for segment in transcription['segments']:
        formatted_transcript += f"[{int(segment['start'])}s] {segment['text'].strip()}\n"

    # Common prompt structure, adaptable for all models
    system_prompt = "You are a precise data extraction tool. Your ONLY purpose is to extract start and end timestamps from a video transcript. You MUST respond with a valid JSON object and NOTHING else. Do NOT add any explanations, introductions, or summaries. If you cannot find suitable clips, you MUST return an empty list."
    user_prompt = f"From the following transcript, extract the {num_clips} most engaging segments. Rules: 1. Each clip's duration must be between {min_duration} and {max_duration} seconds. 2. The \"end\" time for each clip must align with the end of a sentence or a complete thought. 3. The response format must be a JSON object containing a key \"clips\" with a list of objects, where each object has a \"start\" and \"end\" key. Example: {{\"clips\": [{{\"start\": 122, \"end\": 165}}, {{\"start\": 345, \"end\": 398}}]}}. Transcript:\n---\n{formatted_transcript}\n---"

    content = ""
    try:
        if ai_service == "openai":
            response = client.chat.completions.create(model="gpt-4o-mini",
                                                      messages=[{"role": "system", "content": system_prompt},
                                                                {"role": "user", "content": user_prompt}],
                                                      response_format={"type": "json_object"}, temperature=0.2)
            content = response.choices[0].message.content

        elif ai_service == "gemini":
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                ),
            )
            content = response.text
            print(response.text)

        elif ai_service == "ollama":
            response = ollama.chat(model=OLLAMA_MODEL, messages=[{'role': 'system', 'content': system_prompt},
                                                                 {'role': 'user', 'content': user_prompt}],
                                   format='json')
            content = response['message']['content']

        # The parsing function will now sanitize the content before parsing
        return _parse_llm_response(content)

    except Exception as e:
        print(f"CRITICAL ERROR with {ai_service.capitalize()}: {e}")
        return []


class ClipCounter:
    def __init__(self): self.count = 0
    def get(self): self.count += 1; return self.count


def _get_video_id(source: str) -> str:
    """Generates a filesystem-safe ID from a YouTube URL or local file path."""
    if source.startswith(('http://', 'https://')):
        match = re.search(r"v=([a-zA-Z0-9_-]{11})", source)
        if match:
            return match.group(1)
    # For local files, use the filename without extension
    return os.path.splitext(os.path.basename(source))[0]


def create_clip_with_subtitles(original_video_path, chunk_transcription, absolute_start_time, absolute_end_time,
                               clip_start_in_chunk, clip_num):
    """Creates a video clip from the original video using absolute times, but generates subtitles using the chunk's transcription."""
    print(f"Creating clip #{clip_num} from original video at {absolute_start_time}s to {absolute_end_time}s...")

    # ### MODIFICATION: Use 'with' statement for robust resource management ###
    with mp.VideoFileClip(original_video_path) as video:
        with video.subclip(absolute_start_time, absolute_end_time) as clip:
            def subtitle_generator(txt):
                words_in_clip = []
                clip_end_in_chunk = clip_start_in_chunk + (absolute_end_time - absolute_start_time)
                for segment in chunk_transcription['segments']:
                    for word_info in segment['words']:
                        if word_info['start'] >= clip_start_in_chunk and word_info['end'] <= clip_end_in_chunk:
                            relative_start, relative_end = word_info['start'] - clip_start_in_chunk, word_info[
                                'end'] - clip_start_in_chunk
                            words_in_clip.append(((relative_start, relative_end), word_info['word']))
                return words_in_clip

            subtitles = SubtitlesClip(subtitle_generator(chunk_transcription),
                                      lambda txt: mp.TextClip(txt, font='Arial-Bold', fontsize=48, color='white',
                                                              stroke_color='black', stroke_width=2, method='caption',
                                                              size=(clip.w * 0.8, None)))
            final_clip = mp.CompositeVideoClip([clip, subtitles.set_position(('center', 0.75), relative=True)])

            (w, h) = final_clip.size
            if w / h > 9.0 / 16.0:
                final_clip = final_clip.crop(x_center=w / 2, width=int(h * 9.0 / 16.0))

            base_name = os.path.splitext(os.path.basename(original_video_path))[0]
            output_filename = os.path.join(OUTPUT_DIR, f"{base_name}_clip_{clip_num}.mp4")

            final_clip.write_videofile(output_filename, codec="libx264", audio_codec="aac",
                                       temp_audiofile='temp-audio.m4a', remove_temp=True, preset="medium",
                                       bitrate="5000k", threads=4, fps=clip.fps, logger=None)
            print(f"Successfully created clip: {output_filename}")


def process_video_chunk(original_video_path, chunk_path, time_offset, clip_counter, num_clips, min_duration,
                        max_duration):
    """Transcribes a video chunk, finds highlights, and creates clips."""
    print(f"\n--- Processing chunk: {os.path.basename(chunk_path)} (starts at {time_offset:.2f}s) ---")
    try:
        transcription = transcribe_audio(chunk_path)
        if not transcription['segments']: print("No speech detected in this chunk."); return

        if AI_SERVICE:
            highlights = find_highlights_with_llm(transcription, num_clips, min_duration, max_duration, AI_SERVICE)
            if not highlights: print("No highlights found in this chunk."); return
            for h in highlights:
                absolute_start = h['start'] + time_offset;
                absolute_end = h['end'] + time_offset
                create_clip_with_subtitles(original_video_path, transcription, absolute_start, absolute_end, h['start'],
                                           clip_counter.get())
        else:
            print("No AI service is available to find highlights.")
    except Exception as e:
        print(f"An error occurred while processing chunk {chunk_path}: {e}")


def main_workflow(input_source: str, num_clips_per_chunk: int, min_duration: int, max_duration: int,
                  force_process: bool = False):
    """The main function with robust caching and chunking logic for video processing."""
    try:
        video_id = _get_video_id(input_source)
        video_cache_dir = os.path.join(CACHE_DIR, video_id)
        source_cache_path = os.path.join(video_cache_dir, "source.mp4")

        if force_process and os.path.exists(video_cache_dir):
            print("Force processing enabled. Deleting existing cache...")
            shutil.rmtree(video_cache_dir)

        if not os.path.exists(video_cache_dir):
            print(f"No cache found for '{video_id}'. Processing from scratch.")
            os.makedirs(video_cache_dir, exist_ok=True)
            if input_source.startswith(('http://', 'https://')):
                source_cache_path = download_video(input_source, source_cache_path)
            else:
                print(f"Copying local file to cache: {input_source}")
                shutil.copy(input_source, source_cache_path)

            print("Chunking video...")
            # Get duration first without keeping the file open
            with mp.VideoFileClip(source_cache_path) as video:
                duration = video.duration

            if duration <= CHUNK_DURATION:
                shutil.copy(source_cache_path, os.path.join(video_cache_dir, "chunk_0.mp4"))
            else:
                num_chunks = int(-(-duration // CHUNK_DURATION))
                for i in range(num_chunks):
                    start_time = i * CHUNK_DURATION
                    end_time = min((i + 1) * CHUNK_DURATION, duration)
                    chunk_path = os.path.join(video_cache_dir, f"chunk_{i}.mp4")
                    print(f"  - Creating chunk {i+1}/{num_chunks}...")

                    # ### MODIFICATION: Open the video inside the loop for each chunk ###
                    # This is the key fix to prevent the 'NoneType' error.
                    with mp.VideoFileClip(source_cache_path) as video:
                        with video.subclip(start_time, end_time) as chunk_clip:
                            chunk_clip.write_videofile(chunk_path, codec="libx264", preset="fast", logger=None)
        else:
            print(f"Cache found for '{video_id}'. Using cached video and chunks.")

        clip_counter = ClipCounter()
        chunk_paths = sorted(glob.glob(os.path.join(video_cache_dir, "chunk_*.mp4")))
        for i, chunk_path in enumerate(chunk_paths):
            time_offset = i * CHUNK_DURATION
            process_video_chunk(source_cache_path, chunk_path, time_offset, clip_counter, num_clips_per_chunk, min_duration, max_duration)

        print("\nAll processing complete!")

    except Exception as e:
        import traceback
        print(f"\nAn error occurred in the main workflow: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    INPUT_SOURCE = "https://www.youtube.com/watch?v=R0Fxd13ogUg"
    # INPUT_SOURCE = r"E:\TV\Brooklyn Nine Nine\Season 7\Brooklyn.Nine-Nine.S07E06.WEBRip.x264-ION10.mp4"
    NUM_CLIPS_PER_CHUNK = 1
    MIN_CLIP_DURATION = 30
    MAX_CLIP_DURATION = 60
    FORCE_PROCESS = False
    main_workflow(INPUT_SOURCE, NUM_CLIPS_PER_CHUNK, MIN_CLIP_DURATION, MAX_CLIP_DURATION, FORCE_PROCESS)