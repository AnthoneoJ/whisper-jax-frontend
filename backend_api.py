VERSION = '0.0.2'

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from backend import *


app = FastAPI()


@app.post("/infer_audio")
def call_infer_audio(task: str, return_timestamps: str, file: UploadFile = File(...)):
    contents = file.file.read()
    response_data = infer_audio(task, return_timestamps, contents)
    return JSONResponse(content=response_data)

@app.post("/infer_youtube")
def call_infer_youtube(youtube_url:str, task: str, return_timestamps: str):
    response_data = infer_youtube(youtube_url, task, return_timestamps)
    return JSONResponse(content=response_data)