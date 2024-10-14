import cv2
import os

# Create a directory to store images if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define the coordinates of the ROI (Region of Interest)
x, y, width, height = 50, 50, 400, 300

# Counter for image naming
counter = 50

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Crop the frame to the ROI
    roi = frame[y:y+height, x:x+width]

    # Display the resulting frame
    cv2.imshow('Webcam', roi)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF

    # If 'r' is pressed, save the image
    if key == ord('r'):
        filename = f'images/{counter}.jpg'
        cv2.imwrite(filename, roi)
        print(f"Image {counter} saved as {filename}")
        counter += 1

    # If 'q' is pressed, quit
    elif key == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()