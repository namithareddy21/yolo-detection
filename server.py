from fastapi import FastAPI
from pydantic import BaseModel
import base64
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()
model = YOLO("yolov8n.pt")

class ImageData(BaseModel):
    image: str

@app.post("/detect")
async def detect(data: ImageData):
    img_bytes = base64.b64decode(data.image)
    img_array = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    results = model(frame)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        label = results.names[int(box.cls)]
        conf = float(box.conf)
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "label": label,
            "confidence": conf,
            "width": x2 - x1,
            "height": y2 - y1
        })

    return {"detections": detections}


