import cv2
from pathlib import Path

video = cv2.VideoCapture("flight.mp4")

if not video.isOpened():
    raise RuntimeError("Could not open flight.mp4. Check its name and location.")
fps = video.get(cv2.CAP_PROP_FPS)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

print("Frames per second:", fps)
print("Total frames:", frame_count)

if fps > 0:
    print("Duration in seconds:", round(frame_count / fps, 1))
output_folder = Path("frames")
output_folder.mkdir(exist_ok=True)

frame_index = 0
saved_count = 0

while True:
    success, frame = video.read()

    if not success:
        break

    if frame_index % 15 == 0:
        filename = output_folder / f"frame_{frame_index:04d}.jpg"

        if not cv2.imwrite(str(filename), frame):
            raise RuntimeError(f"Could not save {filename}")

        saved_count += 1

    frame_index += 1

video.release()
print(f"Saved {saved_count} images to the frames folder.")
