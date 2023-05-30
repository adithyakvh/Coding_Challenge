from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import matplotlib.pyplot as plt
import copy
import csv
import os
'''
ap = argparse.ArgumentParser()
ap.add_argument("-v", "-D:\\Adithya\\GRE\\Applications\\University of Pennsylvania\\Coding Challenge\\ball_tracking_video.mp4",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
pts = deque(maxlen=args["buffer"])
# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()
# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])
# allow the camera or video file to warm up
time.sleep(2.0)
'''


def write_bounding_boxes_to_csv(file_path, bounding_boxes):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame Number', 'X Center', 'Y Center', 'Width', 'Height'])
        writer.writerows(bounding_boxes)

file_path = r"D:\\Adithya\\GRE\\Applications\\University of Pennsylvania\\Coding Challenge\\Box_coordinates.csv"

def draw_rectangle(img, top_left, bot_right, colour_arr, thickness):
    cv2.rectangle(img, top_left, bot_right, colour_arr, thickness)

def find_max_votes(hsv_image, hough_circles, lower_threshold, upper_threshold):

    max_votes = 0
    roi = None
    roi_center = None
    roi_radius = None
    
    for i in hough_circles[0, :]:
        circle_center = (i[0], i[1])
        r = i[2]
        x1, y1 = np.array(i[0:2]) - r
        x2, y2 = np.array(i[0:2]) + r
        votes = np.where((np.all(lower_threshold.reshape(1, 1, 3) < hsv_image[y1:y2, x1:x2, :], axis=2)) & 
                         (np.all(hsv_image[y1:y2, x1:x2, :] < upper_threshold.reshape(1, 1, 3), axis=2)))
        
        if len(votes[0]) > max_votes:
            max_votes = len(votes[0])
            roi = votes
            roi_center = circle_center
            roi_radius = r

    return max_votes, roi, roi_center, roi_radius
    
pts = deque(maxlen=64)

# Working values
whiteLower = np.array([15, 0, 200])
whiteUpper = np.array([32, 40, 255])

vs = cv2.VideoCapture("D:\\Adithya\\GRE\\Applications\\University of Pennsylvania\\Coding Challenge\\ball_tracking_video.mp4")
# keep looping
fps = vs.get(cv2.CAP_PROP_FPS)
bounding_boxes = []
frame_id = 0
frame_arr = []

while True:
	# grab the current frame
    frame = vs.read()
    # handle the frame from VideoCapture or VideoStream
    frame = frame[1]
    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        break
    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame_id += 1
    # frame = imutils.resize(frame, width=600)
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    blurred = cv2.GaussianBlur(frame, (13, 13), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 4,
                                param1=60, param2=20,
                                minRadius=32, maxRadius=50)
    
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # construct a mask for the color "white", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, whiteLower, whiteUpper)
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
        # print("radius", radius)
        # only proceed if the radius meets a minimum size
        # if radius > 2:
        

        # draw the circle and centroid on the frame,
        # then update the list of tracked points
        
        x, y, w, h = cv2.boundingRect(c)
        x_outer, y_outer = x+w, y+h

        # Check if hough circles are detected, if not detected, just use HSV Filtering
        if circles is not None:

            circles = np.uint16(np.around(circles))
            max_votes, roi, roi_center, roi_radius = find_max_votes(hsv, circles, whiteLower, whiteUpper)
            if max_votes > 0:

                ##################################
                # Here compare with green box, implement it
                ##################################
                
                gb_x1, gb_y1= np.uint16(np.around(np.array(center) - (0.5*roi_radius)))
                gb_x2, gb_y2= np.uint16(np.around(np.array(center) + (0.5*roi_radius))) 

                green_box_votes = np.where((np.all(whiteLower.reshape(1, 1, 3) < hsv[gb_y1:gb_y2, gb_x1:gb_x2, :], axis=2)) & 
                (np.all(hsv[gb_y1:gb_y2, gb_x1:gb_x2, :] < whiteUpper.reshape(1, 1, 3), axis=2)))

                if 0.75 * len(green_box_votes[0]) > max_votes:
                    x1, y1 = np.uint16(np.around(np.array(center) - (0.8*roi_radius)))
                    x2, y2 = np.uint16(np.around(np.array(center) + (0.8*roi_radius))) 
                    bbox_center = center

                    # circle center
                    # cv2.circle(frame, center, 1, (0, 100, 100), 3)
                    # circle outline                        
                    # cv2.circle(frame, center, 5, (0, 0, 255), -1)
                
                else:
                   
                    x1, y1 = np.array(roi_center) - roi_radius  
                    x2, y2 = np.array(roi_center) + roi_radius
                    bbox_center = roi_center
            
            else:
                ################################
                # Just use the green box
                ################################

                x1, y1 = np.uint16(np.around(np.array(center) - 30))
                x2, y2 = np.uint16(np.around(np.array(center) + 30))
                bbox_center = center
            
        
        else:
            x1, y1 = np.uint16(np.around(np.array(center) - 30))
            x2, y2 = np.uint16(np.around(np.array(center) + 30)) 
            bbox_center = center
            # cv2.circle(frame, center, 5, (0, 0, 255), -1)
            

        # cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 140, 150), 3)
    
    else:
        if circles is not None:
            circles = np.uint16(np.around(circles))
            max_votes, roi, bbox_center, bbox_radius = find_max_votes(hsv, circles, whiteLower, whiteUpper)
            if max_votes > 0:
                bbox_found = True


        else:
            pass
    


    if bbox_found:
        # circle center
        cv2.circle(frame, bbox_center, 2, (0, 0, 255), 3)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 140, 150), 3) 
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bounding_boxes.append([frame_id, bbox_center[0], bbox_center[1], bbox_width, bbox_height])


    # cv2.putText(frame, str(frame_id), (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    frame_arr.append(frame) 
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# otherwise, release the camera
if len(bounding_boxes) > 0:
    write_bounding_boxes_to_csv(file_path, bounding_boxes)


vs.release()
# close all windows
cv2.destroyAllWindows()
print("os", os.getcwd())
output_filename = "D:/Adithya/GRE/Applications/University of Pennsylvania/Coding Challenge/ball_tracking.mp4"
out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
for each_frame in frame_arr:
    # Write the frame to the video
    out.write(each_frame)
out.release()
