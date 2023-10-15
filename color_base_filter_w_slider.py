import cv2
import numpy as np

# Initialize low and high HSV values for the color orange
lower_orange = np.array([2, 100, 100])
upper_orange = np.array([75, 255, 255])

def nothing(x):
    pass

frame = cv2.imread("img.png")
frame = cv2.resize(frame, (640, 400))

# Create trackbars for adjusting HSV values
cv2.namedWindow('Trackbars')
cv2.createTrackbar('Low H', 'Trackbars', lower_orange[0], 255, nothing)
cv2.createTrackbar('Low S', 'Trackbars', lower_orange[1], 255, nothing)
cv2.createTrackbar('Low V', 'Trackbars', lower_orange[2], 255, nothing)
cv2.createTrackbar('High H', 'Trackbars', upper_orange[0], 255, nothing)
cv2.createTrackbar('High S', 'Trackbars', upper_orange[1], 255, nothing)
cv2.createTrackbar('High V', 'Trackbars', upper_orange[2], 255, nothing)

# Function to draw contour on the mask
def draw_contour(frame_1, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:  # Check if any contour exists
        largest_contour = max(contours, key=cv2.contourArea)
        contour_with_red_line = cv2.drawContours(frame_1.copy(), [largest_contour], -1, (0, 0, 255), 2)
        contour_area = cv2.contourArea(largest_contour)
        contour_with_red_line = cv2.putText(contour_with_red_line, f"{contour_area}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return contour_with_red_line
    else:
        return frame_1  # If no contour, return the original frame

while True:
    try:
        # Get current trackbar positions
        low_h = cv2.getTrackbarPos('Low H', 'Trackbars')
        low_s = cv2.getTrackbarPos('Low S', 'Trackbars')
        low_v = cv2.getTrackbarPos('Low V', 'Trackbars')
        high_h = cv2.getTrackbarPos('High H', 'Trackbars')
        high_s = cv2.getTrackbarPos('High S', 'Trackbars')
        high_v = cv2.getTrackbarPos('High V', 'Trackbars')
        
        # Make sure upper value is not less than the lower value
        low_h, high_h = min(low_h, high_h), max(low_h, high_h)
        low_s, high_s = min(low_s, high_s), max(low_s, high_s)
        low_v, high_v = min(low_v, high_v), max(low_v, high_v)
        
        # Update HSV values based on trackbar positions
        lower_orange = np.array([low_h, low_s, low_v])
        upper_orange = np.array([high_h, high_s, high_v])
        
        # Apply the orange color filter
        orange_mask = cv2.inRange(frame, lower_orange, upper_orange)
        filtered_frame = cv2.bitwise_and(frame, frame, mask=orange_mask)
        
        # Stack frames
        contour_frame = draw_contour(frame, orange_mask)
        stacked_frame = np.vstack((frame, cv2.cvtColor(orange_mask, cv2.COLOR_GRAY2BGR), contour_frame))
        
        cv2.imshow('Original vs Filtered', stacked_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    except ValueError as e:
        print(f"An error occurred: {e}")

cv2.destroyAllWindows()
