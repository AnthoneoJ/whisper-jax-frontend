BACKEND_VERSION = '0.0.2'

import os, time, tempfile, logging
from multiprocessing import Pool

import numpy as np
import jax.numpy as jnp
from jax.experimental.compilation_cache import compilation_cache as cc
from transformers.pipelines.audio_utils import ffmpeg_read
import yt_dlp as youtube_dl

from whisper_jax import FlaxWhisperPipline


cc.initialize_cache("./jax_cache")
checkpoint = "openai/whisper-large-v3" #"openai/whisper-medium"

BATCH_SIZE = 32
CHUNK_LENGTH_S = 30
NUM_PROC = 32
YT_LENGTH_LIMIT_S = 7200  # limit to 2 hour YouTube files

logger = logging.getLogger("whisper-jax-app")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)

pipeline = FlaxWhisperPipline(checkpoint, dtype=jnp.bfloat16, batch_size=BATCH_SIZE) # use jnp.float16 on small GPU
stride_length_s = CHUNK_LENGTH_S / 6
chunk_len = round(CHUNK_LENGTH_S * pipeline.feature_extractor.sampling_rate)
stride_left = stride_right = round(stride_length_s * pipeline.feature_extractor.sampling_rate)
step = chunk_len - stride_left - stride_right
pool = Pool(NUM_PROC)

# do a pre-compile step so that the first user to use the demo isn't hit with a long transcription time
logger.info("compiling forward call...")
start = time.time()
random_inputs = {
    "input_features": np.ones(
        (BATCH_SIZE, pipeline.model.config.num_mel_bins, 2 * pipeline.model.config.max_source_positions)
    )
}
random_timestamps = pipeline.forward(random_inputs, batch_size=BATCH_SIZE, return_timestamps=True)
compile_time = time.time() - start
logger.info(f"compiled in {compile_time}s")


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
    
def download_yt_audio(yt_url, filename):
    info_loader = youtube_dl.YoutubeDL()
    try:
        info = info_loader.extract_info(yt_url, download=False)
    except youtube_dl.utils.DownloadError as err:
        raise RuntimeError(str(err))

    file_length = info["duration_string"]
    file_h_m_s = file_length.split(":")
    file_h_m_s = [int(sub_length) for sub_length in file_h_m_s]
    if len(file_h_m_s) == 1:
        file_h_m_s.insert(0, 0)
    if len(file_h_m_s) == 2:
        file_h_m_s.insert(0, 0)

    file_length_s = file_h_m_s[0] * 3600 + file_h_m_s[1] * 60 + file_h_m_s[2]
    if file_length_s > YT_LENGTH_LIMIT_S:
        yt_length_limit_hms = time.strftime("%HH:%MM:%SS", time.gmtime(YT_LENGTH_LIMIT_S))
        file_length_hms = time.strftime("%HH:%MM:%SS", time.gmtime(file_length_s))
        raise ValueError(f"Maximum YouTube length is {yt_length_limit_hms}, got {file_length_hms} YouTube video.")

    ydl_opts = {"outtmpl": filename, "format": "worstvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([yt_url])
        except youtube_dl.utils.ExtractorError as err:
            raise RuntimeError(str(err))

def tqdm_generate(inputs: dict, task: str, return_timestamps: bool):
    dataloader = pipeline.preprocess_batch(inputs, chunk_length_s=CHUNK_LENGTH_S, batch_size=BATCH_SIZE)
    logger.info("pre-processing audio file...")
    dataloader = pool.map(identity, dataloader)
    logger.info("done post-processing")

    model_outputs = []
    start_time = time.time()
    logger.info("transcribing...")
    # iterate over our chunked audio samples - always predict timestamps to reduce hallucinations
    for batch in dataloader:
        model_outputs.append(pipeline.forward(batch, batch_size=BATCH_SIZE, task=task, return_timestamps=True))
    runtime = time.time() - start_time
    logger.info("done transcription")

    logger.info("post-processing...")
    post_processed = pipeline.postprocess(model_outputs, return_timestamps=True)
    text = post_processed["text"]
    if return_timestamps:
        timestamps = post_processed.get("chunks")
        timestamps = [
            f"[{format_timestamp(chunk['timestamp'][0])} -> {format_timestamp(chunk['timestamp'][1])}] {chunk['text']}"
            for chunk in timestamps
        ]
        text = "\n".join(str(feature) for feature in timestamps)
    logger.info("done post-processing")
    return text, runtime

def infer_audio(task: str, return_timestamps: str, contents: bytes):
    inputs = ffmpeg_read(contents, pipeline.feature_extractor.sampling_rate)
    inputs = {"array": inputs, "sampling_rate": pipeline.feature_extractor.sampling_rate}
    logger.info("done loading")
    return_timestamps_bool = True if return_timestamps.lower() == "true" else False
    text, runtime = tqdm_generate(inputs, task=task, return_timestamps=return_timestamps_bool)
    response_data = {
        "transcription": text,
        "runtime_seconds": runtime
    }
    return response_data

def infer_youtube(youtube_url:str, task: str, return_timestamps: str):
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = os.path.join(tmpdirname, "video.mp4")
        download_yt_audio(youtube_url, filepath)

        with open(filepath, "rb") as f:
            inputs = f.read()
    
    inputs = ffmpeg_read(inputs, pipeline.feature_extractor.sampling_rate)
    inputs = {"array": inputs, "sampling_rate": pipeline.feature_extractor.sampling_rate}
    logger.info("done loading...")
    return_timestamps_bool = True if return_timestamps.lower() == "true" else False
    text, runtime = tqdm_generate(inputs, task=task, return_timestamps=return_timestamps_bool)
    response_data = {
        "transcription": text,
        "runtime_seconds": runtime
    }
    return response_data