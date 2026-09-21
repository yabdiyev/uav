import cv2

video = cv2.VideoCapture("flight.mp4")

if not video.isOpened():
    raise RuntimeError("Could not open flight.mp4. Check its name and location.")

success, frame = video.read()
video.release()

if not success:
    raise RuntimeError("Opened the video, but could not read its first frame.")

if not cv2.imwrite("first_frame.jpg", frame):
    raise RuntimeError("Could not save the image.")

print("Saved first_frame.jpg")
print("Frame shape:", frame.shape)