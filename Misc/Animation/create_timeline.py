import cv2
import copy
import numpy as np

# Path to the video file
video_path = 'example.mp4'

# Create a VideoCapture object
cap = cv2.VideoCapture(video_path)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Determine the size of the images (resize if necessary)
ret, test_frame = cap.read()
if not ret:
    print("Error: Could not read frame.")
    cap.release()
    exit()

# Resize test frame to determine size
test_frame = cv2.resize(test_frame, None, fx=0.2, fy=0.2)
frame_height, frame_width, _ = test_frame.shape

# Initialize the final canvas to hold all frames side by side
timeline_width = frame_width * 10  # For 10 frames side by side
canvas = np.zeros((frame_height, timeline_width, 3), dtype=np.uint8)

# Define some constants
SLIDER_SPEED    = 1
FRAME_SKIPS     = 0
DIRECTION       = "RIGHT"

# Used to keep track of which frame we are on
frame_count     = 0

# Loop through the video
while cap.isOpened():

    # Read the frame
    ret, frame = cap.read()
    if not ret:
        break

    # For every new frame, shift the canvas to the left and insert new frame column by column
    if frame_count % 15 == 0:  # Adjust as per your skipping logic

        # Get the current frame
        arrow_frame = copy.deepcopy(frame)

        # Resize it before it goes into the canvas
        frame = cv2.resize(frame, None, fx=0.2, fy=0.2)

        # Slowly add the new image to the canvas
        for col in range(0,frame_width,SLIDER_SPEED):
            
            selection_size = SLIDER_SPEED
            # Check if there are enough rows in the image
            if col+selection_size > frame_width:
                selection_size = frame_width-col

            if DIRECTION == "LEFT":
                # Shift canvas to the left
                canvas[:, :-selection_size] = canvas[:, selection_size:]
                # Insert new column from the current frame at the right side
                canvas[:, -selection_size:] = frame[:, col:col+selection_size]  
            else:  # Assuming "RIGHT" direction
                # Shift canvas to the right
                canvas[:, selection_size:] = canvas[:, :-selection_size]
                # Insert new column from the current frame at the left side
                canvas[:, :selection_size] = frame[:, col:col+selection_size][:, ::-1]

            # Make a copy of the canvas
            cv2.imshow("Steering Angle", arrow_frame)

            # Display the canvas
            cv2.imshow('Moving Timeline', canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if DIRECTION == "LEFT":
            # Shift canvas to the left
            canvas[:, :-SLIDER_SPEED] = canvas[:, SLIDER_SPEED:]
            # Insert new column from the current frame at the right side
            canvas[:, -SLIDER_SPEED:] = np.full(np.shape(canvas[:, -SLIDER_SPEED:]), 0)
        else:  # Assuming "RIGHT" direction
            # Shift canvas to the right
            canvas[:, SLIDER_SPEED:] = canvas[:, :-SLIDER_SPEED]
            # Insert new column from the current frame at the left side
            canvas[:, :SLIDER_SPEED] = np.full(np.shape(canvas[:, :SLIDER_SPEED]), 0)

    frame_count += 1

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
