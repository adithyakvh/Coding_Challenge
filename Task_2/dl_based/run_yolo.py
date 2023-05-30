from email.mime import image
from yolov5.detect import run
import numpy as np
from imutils.video import VideoStream


if __name__ == '__main__':
    source = "ball_tracking_video.mp4"
    run(weights='./yolov5/yolov5s.pt', source=source, save_txt=True, classes=32, conf_thres=0.5)
