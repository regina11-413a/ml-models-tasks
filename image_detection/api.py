from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import io

app = FastAPI()
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    im = Image.open(io.BytesIO(image_bytes))
    results = model(im)
    detections = results.pandas().xyxy[0].to_dict(orient="records")
    return {"detections": detections}