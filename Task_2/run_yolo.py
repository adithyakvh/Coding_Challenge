from yolov5.detect import run
import os
import re
import csv
import numpy as np
import cv2
import imutils
from imutils.video import VideoStream


def write_bounding_boxes_to_csv(file_path, bounding_boxes):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame Number', 'X Center', 'Y Center', 'Width', 'Height'])
        writer.writerows(bounding_boxes)

def number_split(s):
    return filter(None, re.split(r'(\d+)', s))

def find_max_votes(hsv_image, bboxes, lower_threshold, upper_threshold):

    max_votes = 0
    roi = None
    roi_center = None
    roi_radius = None
    bbox_x1, bbox_x2 = None, None
    bbox_y1, bbox_y2 = None, None
    for i in bboxes:

        print("bb", type(i[0]))
        x1, y1, x2, y2 = i[1:]
        x1 = np.round(x1).astype(np.uint8)
        y1 = np.round(y1).astype(np.uint8)
        x2 = np.round(x2).astype(np.uint8)
        y2 = np.round(y2).astype(np.uint8)
        circle_center = ((x1+x2)/2, (y1+y2)/2)
        r = (x2 - x1)/2
        
        votes = np.where((np.all(lower_threshold.reshape(1, 1, 3) < hsv_image[y1:y2, x1:x2, :], axis=2)) & 
                         (np.all(hsv_image[y1:y2, x1:x2, :] < upper_threshold.reshape(1, 1, 3), axis=2)))
        
        if len(votes[0]) > max_votes:
            max_votes = len(votes[0])
            roi = votes
            roi_center = circle_center
            roi_radius = r
            bbox_x1, bbox_x2 = x1, x2
            bbox_y1, bbox_y2 = y1, y2

    return max_votes, roi, roi_center, roi_radius, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2)

def hsv_filtering(lower_threshold, upper_threshold, src_path, csv_out_path, video_out_path, image_boxes_dict):
    vs = cv2.VideoCapture(src_path)
    fps = vs.get(cv2.CAP_PROP_FPS)
    bounding_boxes = []
    frame_id = 0
    frame_arr = []

    # keep looping
    while True:
        # grab the current frame
        frame = vs.read()
        # handle the frame from VideoCapture or VideoStream
        frame = frame[1]
        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video
        if frame is None:
            break
        # Blur the frame, and convert it to the HSV color space 
        frame_id += 1

        if frame_id in image_boxes_dict:
            if len(image_boxes_dict[frame_id]) > 1:

                frame_width = frame.shape[1]
                frame_height = frame.shape[0]
                blurred = cv2.GaussianBlur(frame, (13, 13), 0)
                gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
                
                hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
                max_votes, roi, bbox_center, roi_radius, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2) = \
                find_max_votes(hsv, image_boxes_dict[frame_id], lower_threshold, upper_threshold)
                
            else:
                x1, y1, x2, y2 = image_boxes_dict[frame_id][1:]
                
                bbox_x1 = np.round(x1).astype(np.uint8)
                bbox_y1 = np.round(y1).astype(np.uint8)
                bbox_x2 = np.round(x2).astype(np.uint8)
                bbox_y2 = np.round(y2).astype(np.uint8)
                bbox_center = ((bbox_x1 + bbox_x2)/2), ((bbox_y1 + bbox_y2)/2)

                '''
                # construct a mask for the color "white", then perform
                # a series of dilations and erosions to remove any small
                # blobs left in the mask
                mask = cv2.inRange(hsv, lower_threshold, upper_threshold)
                mask = cv2.erode(mask, None, iterations=1)
                mask = cv2.dilate(mask, None, iterations=2)

                # find contours in the mask and initialize the current
                # (x, y) center of the ball
                cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)

                center = None
                bbox_found = False

                # only proceed if at least one contour was found
                if len(cnts) > 0:
                    # find the largest contour in the mask, then use
                    # it to compute the minimum enclosing circle and
                    # centroid
                    bbox_found = True

                    c = max(cnts, key=cv2.contourArea)

                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    M = cv2.moments(c)
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    if radius > 2:
                        x, y, w, h = cv2.boundingRect(c)
                        x_outer, y_outer = x+w, y+h
                        cv2.circle(frame, center, 5, (0, 0, 255), -1)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                '''
            cv2.circle(frame, bbox_center, 2, (0, 0, 255), 3)
            cv2.rectangle(frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (180, 140, 150), 3) 
        
        cv2.imshow("Frame", frame)
        key = cv2.waitKey() & 0xFF
        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break

if __name__ == '__main__':
    source = "D:/Adithya/GRE/Applications/University of Pennsylvania/Coding Challenge/ball_tracking_video.mp4"
    csv_output = r"D:\\Adithya\\GRE\\Applications\\University of Pennsylvania\\Coding Challenge\\Box_coordinates_Yolo.csv"
    video_output = "D:\\Adithya\\GRE\\Applications\\University of Pennsylvania\\Coding Challenge\\ball_tracking_Yolo.mp4"
    lower_threshold = np.array([15, 0, 200])
    upper_threshold = np.array([32, 40, 255])
    run(weights='./yolov5/yolov5s.pt', source=source, save_txt=True, classes=32, conf_thres=0.5)
    all_results_folder = os.path.join('yolov5', 'runs', 'detect')

    image_boxes_dict = {}
    # Check if folder 'exp' exists and rename to exp0
    if os.path.exists(os.path.join(all_results_folder, 'exp')):
        os.rename(os.path.join(all_results_folder, 'exp'), os.path.join(all_results_folder, 'exp0'))
        
    # Get the latest experiment number using from the runs/detect folder
    exp = max([int(re.search('\d+', f.name).group()) for f in os.scandir(all_results_folder) if f.is_dir()])

    # Get the path to the latest experiment folder
    results_folder = os.path.join(all_results_folder, f'exp{exp}', 'labels')

    # iterate over all text files in labels folder
    for file in os.listdir(results_folder):
        # extract bounding boxes and class labels from text file
        id = int(re.findall(r'\d+', file)[-1]) 
        
        with open(os.path.join(results_folder, file), 'r') as f:
            data = f.readlines()
            data = [x.strip() for x in data]
            data = [x.split(' ') for x in data]
        image_boxes_dict[id] = data
    
    # hsv_filtering(lower_threshold, upper_threshold, source, csv_output, video_output, image_boxes_dict)   
