import cv2
import numpy as np

# Define the lower and upper yellow color thresholds in BGR color space

lower_yellow = np.array([0, 0, 100])
upper_yellow = np.array([230, 255, 250])

# Create a video capture object
frame = cv2.imread("mavi3.png")

structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # You can adjust the size as neede

yellow_mask = cv2.inRange(frame, lower_yellow, upper_yellow)
#filtered_frame = cv2.bitwise_and(frame, frame, mask=yellow_mask)


dilated_image = cv2.dilate(yellow_mask, structuring_element, iterations=3)



contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

largest_contour = max(contours, key=cv2.contourArea)

stacked_frame = np.hstack((yellow_mask, dilated_image))
    
cv2.imshow('Original vs Filtered', stacked_frame)

cv2.waitKey(0) & 0xFF == ord('q')
    

cv2.destroyAllWindows()
