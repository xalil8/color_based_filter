import cv2
import numpy as np


######### INITIALIZING LOW AND HIGH VALUES
lower_yellow = np.array([0, 0, 130])
upper_yellow = np.array([166, 255, 255])


frame = cv2.imread("halisaha.png")

frame3 = cv2.imread("photos/mavi1.png")
frame2 = cv2.imread("photos/sari2.png")

def draw_contour(frame_1,mask):

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
        cv2.drawContours(mask, [largest_contour], -1, (0, 0, 255), thickness=2)
        contour_with_red_line = cv2.drawContours(frame_1.copy(), [largest_contour], -1, (0, 0, 255), thickness=2)
        contour_area = cv2.contourArea(largest_contour)
        contour_with_red_line = cv2.putText(contour_with_red_line,f"{contour_area}",(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        return contour_with_red_line


while True:
    # Apply the yellow color filter
    yellow_mask = cv2.inRange(frame, lower_yellow, upper_yellow)
    filtered_frame = cv2.bitwise_and(frame, frame, mask=yellow_mask)
    stacked_frame = np.hstack((frame, cv2.cvtColor(yellow_mask,cv2.COLOR_GRAY2BGR ),draw_contour(frame,yellow_mask)))
    ############## MAIN ########## 

    #########################################
    yellow_mask2 = cv2.inRange(frame2, lower_yellow, upper_yellow)
    filtered_frame2 = cv2.bitwise_and(frame2, frame2, mask=yellow_mask2)
    stacked_frame2 = np.hstack((frame2, cv2.cvtColor(yellow_mask2, cv2.COLOR_GRAY2BGR), draw_contour(frame2, yellow_mask2)))
    #########################################
    yellow_mask3 = cv2.inRange(frame3, lower_yellow, upper_yellow)
    filtered_frame3 = cv2.bitwise_and(frame3, frame3, mask=yellow_mask3)
    stacked_frame3 = np.hstack((frame3, cv2.cvtColor(yellow_mask3, cv2.COLOR_GRAY2BGR), draw_contour(frame3, yellow_mask3)))
    ############## MAIN ########## 
    
    
    common_width = max(stacked_frame2.shape[1], stacked_frame.shape[1], stacked_frame3.shape[1])
    stacked_frame_resized = cv2.resize(stacked_frame, (common_width, stacked_frame.shape[0]))
    stacked_frame2_resized = cv2.resize(stacked_frame2, (common_width, stacked_frame2.shape[0]))
    stacked_frame3_resized = cv2.resize(stacked_frame3, (common_width, stacked_frame3.shape[0]))

    # Vertically stack all three resized frames
    stacked_frames_all = np.vstack((stacked_frame_resized,stacked_frame2_resized, stacked_frame3_resized))
    #stacked_frames_all2 = np.vstack((stacked_frame,stacked_frame0))

    cv2.imshow('Original vs Filtered', stacked_frames_all)
    # Adjust the yellow color thresholds using trackbars
    def nothing(x):
        pass

    ##################TRACKBAR################
    cv2.namedWindow('Trackbars')
    cv2.createTrackbar('Low R', 'Trackbars', lower_yellow[0], 255, nothing)
    cv2.createTrackbar('Low G', 'Trackbars', lower_yellow[1], 255, nothing)
    cv2.createTrackbar('Low B', 'Trackbars', lower_yellow[2], 255, nothing)
    #####
    cv2.createTrackbar('High R', 'Trackbars', upper_yellow[0], 255, nothing)
    cv2.createTrackbar('High G', 'Trackbars', upper_yellow[1], 255, nothing)
    cv2.createTrackbar('High B', 'Trackbars', upper_yellow[2], 255, nothing)
    ##################TRACKBAR################
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    ##############Current trackbar positions########
    low_r = cv2.getTrackbarPos('Low R', 'Trackbars')
    low_g = cv2.getTrackbarPos('Low G', 'Trackbars')
    low_b = cv2.getTrackbarPos('Low B', 'Trackbars')

    high_r = cv2.getTrackbarPos('High R', 'Trackbars')
    high_g = cv2.getTrackbarPos('High G', 'Trackbars')
    high_b = cv2.getTrackbarPos('High B', 'Trackbars')
    ##############Current trackbar positions########

    # Update the lower and upper yellow color thresholds
    lower_yellow = np.array([low_r, low_g, low_b])
    upper_yellow = np.array([high_r, high_g, high_b])

# Release the video capture object and destroy all windows
cv2.destroyAllWindows()
