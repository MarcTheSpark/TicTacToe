import cv2
import numpy as np

# Load timestamps
with open("logForLongVid.txt") as file:
    timestamps = [float(line.strip()) for line in file]

# Create a video writer object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('countdownVid.mp4', fourcc, 30.0, (1920,1080))

# Set font scale and font
font_scale = 1.2 # Increase or decrease based on your preference
font = cv2.FONT_HERSHEY_SIMPLEX

# Loop through timestamps
for i in range(len(timestamps)-1):
    countdown = int(timestamps[i+1] - timestamps[i])
    for j in range(countdown):
        # Create a frame with countdown text
        frame = np.zeros((1080,1920,3), np.uint8)
        cv2.putText(frame, 
            f"Countdown to next musical texture: {countdown-j-1} seconds", 
            (50, 240), 
            font, 
            font_scale, 
            (255, 255, 255), 
            thickness=1, 
            lineType=cv2.LINE_AA)
        # Write the same frame 30 times
        for _ in range(30):
            out.write(frame)

# Release video writer
out.release()

