VERSION = '0.0.0'

import os, logging
from io import BytesIO

import requests
import gradio as gr
from fastapi import UploadFile

from backend_api import infer_audio, infer_youtube


checkpoint = "openai/whisper-large-v3"

API_URL = os.environ.get('API_URL', "http://0.0.0.0:8000")
FILE_LIMIT_MB = 1000

title = "Whisper JAX: The Fastest Whisper API ⚡️"

description = """Whisper JAX is an optimised implementation of the [Whisper model](https://huggingface.co/openai/whisper-large-v3) by OpenAI. It runs on JAX with a TPU v4-8 in the backend. Compared to PyTorch on an A100 GPU, it is over [**70x faster**](https://github.com/sanchit-gandhi/whisper-jax#benchmarks), making it the fastest Whisper API available.

Note that at peak times, you may find yourself in the queue for this demo. When you submit a request, your queue position will be shown in the top right-hand side of the demo pane. Once you reach the front of the queue, your audio file will be transcribed, with the progress displayed through a progress bar.
"""

article = "Whisper large-v3 model by OpenAI. FastAPI backend running JAX on a private GPU machine. Whisper JAX [code](https://github.com/sanchit-gandhi/whisper-jax) and Gradio demo by 🤗 Hugging Face."
logger = logging.getLogger("whisper-jax-app")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)


def identity(batch):
    return batch


# Copied from https://github.com/openai/whisper/blob/c09a7ae299c4c34c5839a76380ae407e7d785914/whisper/utils.py#L50
def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = "."):
    if seconds is not None:
        milliseconds = round(seconds * 1000.0)

        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000

        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000

        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000

        hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
        return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    else:
        # we have a malformed timestamp so just return it as is
        return seconds


if __name__ == "__main__":
    print(f'FRONTEND VERSION: {VERSION}')

    def transcribe_chunked_audio(inputs, task, return_timestamps, progress=gr.Progress()):
        progress(0, desc="Loading audio file...")
        logger.info("loading audio file...")
        if inputs is None:
            logger.warning("No audio file")
            raise gr.Error("No audio file submitted! Please upload an audio file before submitting your request.")
        file_size_mb = os.stat(inputs).st_size / (1024 * 1024)
        if file_size_mb > FILE_LIMIT_MB:
            logger.warning("Max file size exceeded")
            raise gr.Error(
                f"File size exceeds file size limit. Got file of size {file_size_mb:.2f}MB for a limit of {FILE_LIMIT_MB}MB."
            )
        
        with open(inputs, "rb") as f:
            inputs = f.read()
        
        progress(0, desc="Transcribing...")
        file_like = BytesIO(inputs)
        upload_file = UploadFile(filename="example.wav", file=file_like)
        response = infer_audio(task, str(return_timestamps), upload_file)
        if response.status_code == 200:
            res_json = response.json()
            return res_json['transcription'], res_json['runtime_seconds']
        else:
            raise gr.Error(response.text, 0)

    def _return_yt_html_embed(yt_url: str):
        video_id = yt_url.split("?v=")[-1]
        HTML_str = (
            f'<center> <iframe width="500" height="320" src="https://www.youtube.com/embed/{video_id}"> </iframe>'
            " </center>"
        )
        return HTML_str

    def transcribe_youtube(yt_url, task, return_timestamps, progress=gr.Progress()):
        progress(0, desc="Loading audio file...")
        logger.info("loading youtube file...")
        html_embed_str = _return_yt_html_embed(yt_url)

        progress(0, desc="Transcribing...")
        response = infer_youtube(yt_url, task, str(return_timestamps))
        if response.status_code == 200:
            res_json = response.json()
            return html_embed_str, res_json['transcription'], res_json['runtime_seconds']
        else:
            raise gr.Error(response.text, 0)
        
    def update_api(new_api_url):
        global API_URL
        try:
            API_URL = new_api_url
            return f'API_URL successfully updated to {new_api_url}'
        except Exception as e:
            return f"Error: {str(e)}"

    microphone_chunked = gr.Interface(
        fn=transcribe_chunked_audio,
        inputs=[
            gr.Audio(sources=["microphone"], type="filepath"),
            gr.Radio(["transcribe", "translate"], label="Task", value="transcribe"),
            gr.Checkbox(value=False, label="Return timestamps"),
        ],
        outputs=[
            gr.Textbox(label="Transcription", show_copy_button=True),
            gr.Textbox(label="Transcription Time (s)"),
        ],
        allow_flagging="never",
        title=title,
        description=description,
        article=article,
    )

    audio_chunked = gr.Interface(
        fn=transcribe_chunked_audio,
        inputs=[
            gr.Audio(sources=["upload"], label="Audio file", type="filepath"),
            gr.Radio(["transcribe", "translate"], label="Task", value="transcribe"),
            gr.Checkbox(value=False, label="Return timestamps"),
        ],
        outputs=[
            gr.Textbox(label="Transcription", show_copy_button=True),
            gr.Textbox(label="Transcription Time (s)"),
        ],
        allow_flagging="never",
        title=title,
        description=description,
        article=article,
    )

    youtube = gr.Interface(
        fn=transcribe_youtube,
        inputs=[
            gr.Textbox(lines=1, placeholder="Paste the URL to a YouTube video here", label="YouTube URL"),
            gr.Radio(["transcribe", "translate"], label="Task", value="transcribe"),
            gr.Checkbox(value=False, label="Return timestamps"),
        ],
        outputs=[
            gr.HTML(label="Video"),
            gr.Textbox(label="Transcription", show_copy_button=True),
            gr.Textbox(label="Transcription Time (s)"),
        ],
        allow_flagging="never",
        title=title,
        examples=[["https://www.youtube.com/watch?v=m8u-18Q0s7I", "transcribe", False]],
        cache_examples=False,
        description=description,
        article=article,
    )

    settings_tab = gr.Interface(
        fn=update_api,
        inputs=[
            gr.Textbox(API_URL, label="API URL:", interactive=True),
        ],
        outputs=[
            gr.Textbox(label="Result"),
        ],
        allow_flagging="never",
        title=title,
        description=f"UI v{VERSION}",
        article=article,
    )

    demo = gr.Blocks()

    with demo:
        gr.TabbedInterface([microphone_chunked, audio_chunked, youtube, settings_tab], 
                           ["Microphone", "Audio File", "YouTube", "Settings"])

    demo.queue(max_size=5)
    demo.launch(share=True, server_name="0.0.0.0", show_api=False)
