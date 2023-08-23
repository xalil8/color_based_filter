import os
import torch
import cv2
import math
import numpy as np
from ssl import _create_unverified_context
from time import time
from collections import defaultdict, deque
import telegram


class SpeedTracker:
    def __init__(self, bot_token, chat_id, source_video_path, video_saving_path, model_path, writer):
        _create_default_https_context = _create_unverified_context
        self.writer = writer


        self.lower_yellow = np.array([0, 0, 85])
        self.upper_yellow = np.array([230, 255, 255])

        self.last_photo_sent_time = 0
        self.temp_time = {}  # Initialize the car_no_change_count dictionary
        self.source_video_path = source_video_path
        self.video_saving_path = video_saving_path

        self.bot = telegram.Bot(token=bot_token)
        self.chat_id = chat_id

        self.video_cap = cv2.VideoCapture(self.source_video_path)
        width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


        self.bot.send_message(chat_id=self.chat_id,text="sasa exmaple code started ")

        if writer:
            self.result = cv2.VideoWriter(video_saving_path, cv2.VideoWriter_fourcc(*'mp4v'), 16, (width, height))

        #  DETECTION MODEL
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.model = torch.hub.load("ultralytics/yolov5", "yolov5m", force_reload=False, device=device)
        self.model = torch.hub.load("ultralytics/yolov5", "custom",path=model_path, force_reload=False, device="mps")
        
        #self.names = self.model.names
        self.model.conf = 0.8
        self.names = self.model.names


    def reconnect_video(self, video_cap):
        video_cap.release()
        video_cap = cv2.VideoCapture(self.source_video_path)
        return video_cap
    
    def draw_contour(self,frame,x1,y1,x2,y2):
        roi = frame[y1:y2, x1:x2]
        yellow_mask = cv2.inRange(roi,self.lower_yellow, self.upper_yellow)
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            yellow_mask = cv2.cvtColor(yellow_mask,cv2.COLOR_GRAY2BGR)
            cv2.drawContours(yellow_mask, [largest_contour], -1, (0, 0, 255), thickness=2)
            contour_with_red_line = cv2.drawContours(roi, [largest_contour], -1, (255, 0, 255), thickness=2)
            contour_area = cv2.contourArea(largest_contour)
            return yellow_mask,contour_area
        
    def process(self):
        count = 0
        prev_time = time()
        while self.video_cap.isOpened():
            try:
                ret, frame = self.video_cap.read()
                frame = cv2.resize(frame,(1920,1080))
                if not ret:
                    raise Exception("Error reading frame")
                count += 1
                if count % 1 != 0:
                    continue

                # curr_time = time()
                # elapsed_time = curr_time - prev_time # elapsed_time is time that frame takes 
                # prev_time = curr_time
                # fps = 1.0 / elapsed_time
                # cv2.putText(frame, f"FPS: {int(fps)}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 229, 204), 3)

                results = self.model(frame)
                det = results.xyxy[0]


                if det is not None and len(det):
                    for j, (output) in enumerate(det):
                        bbox = output[0:4]
                        conf = round(float(output[4]),2)
                        id = int(output[5])
                        x1, y1, x2, y2 = bbox
                        x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
                        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

                        if id == 1: 
                            filtered_image, contour_area  =self.draw_contour(frame,x1,y1,x2,y2)
                            #print(contour_area)
                            cv2.rectangle(frame, (x1, y1), (x2, y2),  (0, 255, 255), 2)
                            cv2.circle(frame, (center_x, center_y), radius=3, color=(0, 255, 255), thickness=-1)
                            cv2.putText(frame, f"Ziyaretçi  {conf}", (x1+10 , y1-20),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                            current_time = time()

                            # time_since_last_photo = current_time - self.last_photo_sent_time
                            # if time_since_last_photo >= 60:  # 60 seconds
                            #     resized = cv2.resize(frame, (1280, 720))
                            #     cv2.imwrite("yellow.jpg", resized)
                            #     cv2.imwrite("contour_image.jpg", filtered_image)
                            #     self.bot.send_photo(chat_id=self.chat_id, photo=open("contour_image.jpg", 'rb'), caption=f"contour area {contour_area}")
                            #     self.bot.send_photo(chat_id=self.chat_id, photo=open("yellow.jpg", 'rb'), caption="Yabancı Kişi İhlal")
                            #     self.last_photo_sent_time = current_time  # Update the last photo sent time


                if not self.writer:
                    resized = cv2.resize(frame,(1500,1200))
                    cv2.imshow("ROI", resized)
                    if cv2.waitKey(13) == ord('q'):
                        break
                    
                # if self.writer:
                #     #print(f"frame {count} writing")
                #     #cv2.imshow("ROI", frame)
                #     self.result.write(frame)

            except Exception as e:
                print(f"Error: {str(e)}")
                print("Reconnecting to video source...")
                self.video_cap = self.reconnect_video(self.video_cap)  # Update variable name to self.video_cap

        # Release resources
        self.video_cap.release()
        if self.writer:
            self.result.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    writer = False

    bot_token = ""
    chat_id = ""

    source_video_path="v1.mp4"
    video_saving_path = "try_except.mp4"

    model_path = "v4.pt"

    tracker = SpeedTracker(bot_token, chat_id, source_video_path, video_saving_path, model_path,writer)
    tracker.process()
