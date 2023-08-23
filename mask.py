import cv2
import numpy as np



def draw_contour(frame_1,mask):

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour (assuming it's the yellow region)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Draw the largest contour in red on top of the yellow mask
        mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
        cv2.drawContours(mask, [largest_contour], -1, (0, 0, 255), thickness=2)

        contour_with_red_line = cv2.drawContours(frame_1.copy(), [largest_contour], -1, (0, 0, 255), thickness=2)
        
        # Get the area of the largest contour
        #contour_area = cv2.contourArea(largest_contour)

        return contour_with_red_line

# Define the lower and upper yellow color thresholds in BGR color space
lower_yellow = np.array([0, 0, 110])
upper_yellow = np.array([230, 255, 250])

# Create a video capture object
frame = cv2.imread("mavi1.png")

# Create the yellow mask
yellow_mask = cv2.inRange(frame, lower_yellow, upper_yellow)


a = draw_contour(frame,yellow_mask)

cv2.imshow('Yellow Mask with Red Contour', a)

# Print the area of the largest contour
#print(f"Largest contour area: {contour_area}")



cv2.waitKey(0) & 0xFF == ord('q')

cv2.destroyAllWindows()
