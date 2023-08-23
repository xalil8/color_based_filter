import cv2
import numpy as np

# Define the lower and upper blue color thresholds in BGR color space
lower_blue = np.array([0, 0, 0])
upper_blue = np.array([180, 100, 100])

# Create a video capture object
frame = cv2.imread("mavi1.png")
while True:
    # Apply the blue color filter
    blue_mask = cv2.inRange(frame, lower_blue, upper_blue)
    filtered_frame = cv2.bitwise_and(frame, frame, mask=blue_mask)

    # Display the original and filtered frames
    stacked_frame = np.hstack((frame, filtered_frame))
    
    cv2.imshow('Original vs Filtered', stacked_frame)

    # Adjust the blue color thresholds using trackbars
    def nothing(x):
        pass

    cv2.namedWindow('Trackbars')
    cv2.createTrackbar('Low B', 'Trackbars', lower_blue[0], 255, nothing)
    cv2.createTrackbar('High B', 'Trackbars', upper_blue[0], 255, nothing)

    # Wait for a key press and exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Get the current trackbar positions
    low_b = cv2.getTrackbarPos('Low B', 'Trackbars')
    high_b = cv2.getTrackbarPos('High B', 'Trackbars')

    # Update the lower and upper blue color thresholds
    lower_blue = np.array([low_b, 0, 0])
    upper_blue = np.array([high_b, 100, 100])

# Release the video capture object and destroy all windows
cv2.destroyAllWindows()
