import cv2
import numpy as np
import tempfile

def process_video(video_bytes):
  frames = []

# Save bytes to temp file
with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
  temp.write(video_bytes)
  temp_path = temp.name

cap = cv2.VideoCapture(temp_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_skip = max(fps // 10, 1) # target ~10 FPS

count = 0

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

if count % frame_skip == 0:
  frame = cv2.resize(frame, (256, 256))
  frames.append(frame)

count += 1

cap.release()
return frames
