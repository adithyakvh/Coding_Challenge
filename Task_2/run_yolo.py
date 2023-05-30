from email.mime import image
from yolov5.detect import run
import numpy as np
from imutils.video import VideoStream


if __name__ == '__main__':
    source = "D:/Adithya/GRE/Applications/University of Pennsylvania/Coding Challenge/ball_tracking_video.mp4"
    csv_output = r"D:\\Adithya\\GRE\\Applications\\University of Pennsylvania\\Coding Challenge\\Box_coordinates_Yolo.csv"
    video_output = "D:\\Adithya\\GRE\\Applications\\University of Pennsylvania\\Coding Challenge\\ball_tracking_Yolo.mp4"
    run(weights='./yolov5/yolov5s.pt', source=source, save_txt=True, classes=32, conf_thres=0.5)
