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

# This line is specific to your system and can be kept.
cf.IMAGEMAGICK_BINARY = r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"

# --- Configuration ---
OUTPUT_DIR = "output"
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


# --- Core Functions (download_video, transcribe_audio are unchanged) ---
# def download_video(url: str) -> str:
#     # This function remains unchanged.
#     print(f"Downloading video from URL: {url}")
#     if not shutil.which("ffmpeg"): raise RuntimeError("ffmpeg not found.")
#     output_template = os.path.join(OUTPUT_DIR, '%(title)s.%(ext)s')
#     ydl_opts = {'format': 'bestvideo+bestaudio/best', 'merge_output_format': 'mp4', 'outtmpl': output_template,
#                 'verbose': False, 'nocheckcertificate': True, 'hls_prefer_ffmpeg': True, 'quiet': True}
#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         try:
#             ydl.cache.remove()
#         except Exception:
#             pass
#         info = ydl.extract_info(url, download=True)
#         info['ext'] = 'mp4';
#         filename = ydl.prepare_filename(info)
#         if not os.path.exists(filename) or os.path.getsize(filename) == 0:
#             raise yt_dlp.utils.DownloadError("File was not created or is empty.")
#         print(f"Video downloaded to: {filename}")
#         return filename


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
            # chat = client.chats.create(model='gemini-2.0-flash-001')
            # response = chat.send_message([{"role": "system", "content": system_prompt},
            #                                                     {"role": "user", "content": user_prompt}])
            # content = response.text

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

        # Unified parsing logic
        print(f"LLM Raw Response: {content}")
        data = json.loads(content)
        highlights = []
        if isinstance(data, dict) and 'clips' in data and isinstance(data['clips'], list):
            highlights = data['clips']
        elif isinstance(data, list):
            highlights = data
        elif isinstance(data, dict) and 'start' in data and 'end' in data:
            highlights = [data]
        if not highlights or not all('start' in d and 'end' in d for d in highlights):
            raise ValueError("LLM did not return the expected format of clip objects.")
        print(f"{ai_service.capitalize()} identified highlights: {highlights}")
        return highlights

    except Exception as e:
        print(f"CRITICAL ERROR with {ai_service.capitalize()}: {e}")
        return []


def create_clip_with_subtitles(original_video_path, chunk_transcription, absolute_start_time, absolute_end_time,
                               clip_start_in_chunk, clip_num):
    # This function remains unchanged.
    print(f"Creating clip #{clip_num} from original video at {absolute_start_time}s to {absolute_end_time}s...")
    video = mp.VideoFileClip(original_video_path)
    clip = video.subclip(absolute_start_time, absolute_end_time)

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
    if w / h > 9.0 / 16.0: final_clip = final_clip.crop(x_center=w / 2, width=int(h * 9.0 / 16.0))
    base_name = os.path.splitext(os.path.basename(original_video_path))[0]
    output_filename = os.path.join(OUTPUT_DIR, f"{base_name}_clip_{clip_num}.mp4")
    final_clip.write_videofile(output_filename, codec="libx264", audio_codec="aac", temp_audiofile='temp-audio.m4a',
                               remove_temp=True, preset="medium", bitrate="5000k", threads=4, fps=video.fps,
                               logger=None)
    print(f"Successfully created clip: {output_filename}")
    video.close()


def process_video_chunk(original_video_path, chunk_path, time_offset, clip_counter, num_clips, min_duration,
                        max_duration):
    """Transcribes a video chunk, finds highlights, and creates clips."""
    print(f"\n--- Processing chunk starting at {time_offset:.2f} seconds ---")
    try:
        transcription = transcribe_audio(chunk_path)
        if not transcription['segments']: print("No speech detected in this chunk."); return

        if AI_SERVICE:
            highlights_in_chunk = find_highlights_with_llm(transcription, num_clips, min_duration, max_duration,
                                                           AI_SERVICE)
            if not highlights_in_chunk: print("No highlights found in this chunk."); return
            for h in highlights_in_chunk:
                absolute_start = h['start'] + time_offset;
                absolute_end = h['end'] + time_offset
                create_clip_with_subtitles(original_video_path, transcription, absolute_start, absolute_end, h['start'],
                                           clip_counter.get())
        else:
            print("No AI service is available to find highlights.")

    except Exception as e:
        print(f"An error occurred while processing chunk {chunk_path}: {e}")
    finally:
        if chunk_path != original_video_path and os.path.exists(chunk_path):
            print(f"Cleaning up temporary chunk file: {chunk_path}");
            os.remove(chunk_path)


def main_workflow(input_source, num_clips_per_chunk, min_duration, max_duration):
    # This function remains unchanged.
    try:
        if os.path.exists(input_source):
            video_path = input_source
        elif input_source.startswith(('http://', 'https://')):
            video_path = download_video(input_source)
        else:
            raise FileNotFoundError(f"Input '{input_source}' is not a valid URL or local file path.")

        with mp.VideoFileClip(video_path) as video:
            duration = video.duration
        clip_counter = ClipCounter()

        if duration <= CHUNK_DURATION:
            process_video_chunk(video_path, video_path, 0, clip_counter, num_clips_per_chunk, min_duration,
                                max_duration)
        else:
            print(f"Video is longer than 5 minutes ({duration:.2f}s). Splitting into chunks.")
            num_chunks = int(-(-duration // CHUNK_DURATION))
            for i in range(num_chunks):
                start_time = i * CHUNK_DURATION;
                end_time = min((i + 1) * CHUNK_DURATION, duration)
                chunk_path = os.path.join(OUTPUT_DIR, f"temp_chunk_{i}.mp4")
                print(f"\n--- Creating chunk {i + 1}/{num_chunks} ({start_time:.2f}s to {end_time:.2f}s) ---")
                with mp.VideoFileClip(video_path) as video:
                    chunk_clip = video.subclip(start_time, end_time)
                    chunk_clip.write_videofile(chunk_path, codec="libx264", preset="fast", logger=None)
                process_video_chunk(video_path, chunk_path, start_time, clip_counter, num_clips_per_chunk, min_duration,
                                    max_duration)
        print("\nAll processing complete!")
    except Exception as e:
        import traceback;
        print(f"\nAn error occurred in the main workflow: {e}");
        traceback.print_exc()


if __name__ == '__main__':
    INPUT_SOURCE = "https://www.youtube.com/watch?v=R0Fxd13ogUg"
    # INPUT_SOURCE = r"E:\TV\Brooklyn Nine Nine\Season 7\Brooklyn.Nine-Nine.S07E06.WEBRip.x264-ION10.mp4"
    NUM_CLIPS_PER_CHUNK = 1
    MIN_CLIP_DURATION = 30
    MAX_CLIP_DURATION = 60
    main_workflow(INPUT_SOURCE, NUM_CLIPS_PER_CHUNK, MIN_CLIP_DURATION, MAX_CLIP_DURATION)