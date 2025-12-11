# server.py

import base64
import json
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app = FastAPI()

# Allow any frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load small GPU/CPU-compatible model
model = YOLO("yolov8n.pt")


@app.get("/")
def root():
    return {"status": "YOLOv8 API running"}


def decode_image(base64_string):
    base64_string = base64_string.split(",")[1]  # remove header
    img_bytes = base64.b64decode(base64_string)
    img_array = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("Client connected.")

    try:
        while True:
            data = await ws.receive_text()
            data = json.loads(data)

            img = decode_image(data["image"])

            results = model(img, verbose=False)[0]

            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                detections.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "conf": conf,
                    "label": label
                })

            await ws.send_text(json.dumps({"detections": detections}))

    except WebSocketDisconnect:
        print("Client disconnected.")
