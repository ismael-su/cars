# type this command in the terminal to run:
# python yolo_vehicle_speed_estimation.py -i pathtoinputvideo -o pathtooutput -y pretrainedmodel

# import necessary library
from support.centroidtracker import CentroidTracker
from support.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from scipy import spatial
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import math
import os
import random
from utils import estimateSpeed, displayVehicleCount, displayFPS, drawDetectionBoxes, boxInPreviousFrames, \
    count_vehicles, midPoint, dist, rect_distance, rectangleCenter, calibrated_dist, isclose
import csv
import pandas as pd
from array import array
from datetime import datetime
from sqlalchemy import create_engine
import sqlite3 as db

# construct the argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input video")
ap.add_argument("-o", "--output", required=True,
                help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
                help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applyong non-maxima suppression")
ap.add_argument("-s", "--skip-frames", type=int, default=2,
                help="# of skip frames between detections")

args = vars(ap.parse_args())


# PURPOSE: Initializing the video writer with the output video path and the same number
# of fps, width and height as the source video 
# PARAMETERS: Width of the source video, Height of the source video, the video stream
# RETURN: The initialized video writer
def initializeVideoWriter(video_width, video_height, videoStream):
    # Getting the fps of the source video
    sourceVideofps = videoStream.get(cv2.CAP_PROP_FPS)
    # initialize our video writer
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    return cv2.VideoWriter(args["output"], fourcc, sourceVideofps,
                           (video_width, video_height), True)


# to check if objects are close or not
angle_factor = 10.0
H_zoom_factor = 0.5


def dist(c1, c2):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def T2S(T):
    S = abs(T / ((1 + T ** 2) ** 0.5))
    return S


def T2C(T):
    C = abs(1 / ((1 + T ** 2) ** 0.5))
    return C


def isclose(p1, p2):
    c_d = dist(p1[2], p2[2])
    if (p1[1] < p2[1]):
        a_w = p1[0]
        a_h = p1[1]
    else:
        a_w = p2[0]
        a_h = p2[1]

    T = 0
    try:
        T = (p2[2][1] - p1[2][1]) / (p2[2][0] - p1[2][0])
    except ZeroDivisionError:
        T = 1.633123935319537e+16
    S = T2S(T)
    C = T2C(T)
    d_hor = C * c_d
    d_ver = S * c_d
    vc_calib_hor = a_w * 1.3
    vc_calib_ver = a_h * 0.4 * angle_factor
    c_calib_hor = a_w * 1.7
    c_calib_ver = a_h * 0.2 * angle_factor
    # print(p1[2], p2[2],(vc_calib_hor,d_hor),(vc_calib_ver,d_ver))
    if (0 < d_hor < vc_calib_hor and 0 < d_ver < vc_calib_ver):
        return 1
    elif 0 < d_hor < c_calib_hor and 0 < d_ver < c_calib_ver:
        return 2
    else:
        return 0


# Setting the threshold for the number of frames to search a vehicle for
FRAMES_BEFORE_CURRENT = 10
inputWidth, inputHeight = 416, 416

# All these classes will be counted as 'vehicles'
list_of_vehicles = ["bicycle", "car", "motorbike", "bus", "truck", "train"]

WIDTH = 1280
HEIGHT = 720

# load the COCO class labels
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# Initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

# init a list of color for different objects
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector and output layer names
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream
if not args.get("input", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(args["input"])
    video_width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
fs = vs.get(cv2.CAP_PROP_FPS)

# Initialization
previous_frame_detections = [{(0, 0): 0} for i in range(FRAMES_BEFORE_CURRENT)]
num_frames, vehicle_count = 0, 0
writer = initializeVideoWriter(video_width, video_height, vs)
start_time = int(time.time())
FR = 0
(W, H) = (None, None)

writer = None
(W, H) = (None, None)
refObj = None

# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

# init centroid tracker
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
# trackers = []
trackableOjects = {}

totalFrames = 0
fps = FPS().start()

while True:

    rects = []
    # num_frames = frameCounter #######
    num_frames += 1
    print("FRAME:\t", num_frames)
    # Initialization for each iteration
    boxes, confidences, classIDs = [], [], []
    cx, cy = {}, {}
    vehicle_crossed_line_flag = False

    # Calculating fps each second
    s_time, numFrames = displayFPS(start_time, num_frames)

    # read the next frame from the file
    (grabbed, frame) = vs.read()
    print("")

    # if the frame was not grabbed, then we have reached the end of the stream
    if not grabbed:
        break

    # resize the frame to have maximum width of 500 pixels
    frame = imutils.resize(frame, width=1280)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # init a writer to write video to disk
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

    # init the status for detecting or tracking
    status = "Waiting"
    rects = []

    # Check to see if we should run a more detection method to aid our tracker
    # if totalFrames % args["skip_frames"] == 0:
    # set the status and init our new set of object trackers
    status = "Detecting"
    trackers = []

    # convert the frame to a blob and pass the blob through the network and obtain the detections
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    # print(time)
    # init ourlists of detected bboxes, confidences, class IDs
    boxes = []
    confidences = []
    classIDs = []
    color = (0, 255, 0)
    # cv2.line(frame, (100, 100), color)
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the classID and confidence of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if LABELS[classID] in list_of_vehicles:

                # filter out weak detections
                if confidence > args["confidence"]:
                    # scale the bboxes back relative to the size of the image
                    # YOLO return the center (x, y) and width, height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center to derive the bottom and left corner of the bboxes
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update bboxes, confidences, classIDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

                    # Save detection time 
                    nowDate = str(datetime.now().date())
                    nowTime = str(datetime.now().time())
                    now = nowTime + ' / ' + nowDate

    # apply non-maxima suppresion to suppress weak
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    # Display Vehicle Count if a vehicle has been detected
    # displayVehicleCount(frame, vehicle_count)
    # ensure at least one detection exists
    if len(idxs) > 0:
        stat = list()
        close_pair = list()
        s_close_pair = list()
        center = list()
        dist = list()

        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # init rect for tracking
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            center.append([int(x + w / 2), int(y + h / 2)])
            stat.append(0)

            startX = boxes[i][0]
            startY = boxes[i][1]
            endX = boxes[i][0] + boxes[i][2]
            endY = boxes[i][1] + boxes[i][3]
            # distt = dist(center[0][0], center[0][1], center[1][0], center[1][1])
            # print("dist", boxes[i+1][1])

            # construct a dlib rectangle object and start dlib correlation tracker
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(startX, startY, endX, endY)
            tracker.start_track(rgb, rect)

            # add the tracker to our list and we can use it during skip frames
            trackers.append(tracker)

    # otherwise, we should use object trackers to estimate speed and obtain a higher frame processing
    # else:
    # loop over the trackers
    for tracker in trackers:

        # set the status
        status = "Tracking"

        # update the tracker and grab the updated position
        tracker.update(rgb)
        pos = tracker.get_position()

        # unpack the position object
        startX = int(pos.left())
        startY = int(pos.top())
        endX = int(pos.right())
        endY = int(pos.bottom())

        x_bar = startX + 0.5 * endX
        y_bar = startY + 0.5 * endY

        if ((startX <= x_bar <= (startX + endX)) and (startY <= y_bar <= (startY + endY))):
            matchCarID = tracker

        # calculate pixel per meter (ppm) based on width and heigh
        # ppm = math.sqrt(math.pow(endX - startX, 2) + math.pow(endY - startY, 2)) / math.sqrt(5)
        # ppm based on width of car
        # ppm = math.sqrt(math.pow(endX-startX, 2))

        # tracking rect
        drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame, COLORS, LABELS)
        # cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # add the bbox coordinates to the rectangles list
        rects.append((startX, startY, endX, endY))

    # vehicle_count, current_detections = count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame, LABELS, list_of_vehicles)

    # Display Vehicle Count if a vehicle has passed the line 
    # displayVehicleCount(frame, vehicle_count)

    # use the centroid tracker to associate the object 1 and object 2
    objects = ct.update(rects)
    # loop over the tracked objects
    speed = 0
    dist = 0
    for (objectID, centroid) in objects.items():
        # init speed array
        # speed = 0
        # print(len(centroid))
        # check to see if a tracktable object exists for the current objectID
        to = trackableOjects.get(objectID, None)

        # if there is no tracktable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)
            # print(to.centroids)
        # otherwise, use it for speed estimation
        else:
            to.centroids.append(centroid)
            # print(len(to.centroids))
            location1 = to.centroids[-2]
            location2 = to.centroids[-1]
            location3 = to.centroids[0]
            location4 = to.centroids[1]
            # location5 = to.centroids[-3]
            # print(to.centroids)
            print(location1, location2, location3, location4)
            # print(to.centroids[-2], to.centroids[-1], to.centroids[0], to.centroids[1])

            # print("location1: ",location1, "location2", location2, "location3", location3, "location4", location4)
            speed = estimateSpeed(location1, location2, location3, location4)
        trackableOjects[objectID] = to

        # cv2.putText(frame, "{:.1f} m".format(dist), (centroid[0]-25,centroid[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        cv2.putText(frame, "{:.1f} km/h".format(speed), (centroid[0] - 15, centroid[1] + 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 0), 2)

    # Draw detection box 
    classIDsS, confidencesS, speeds = drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame, COLORS, LABELS)
    vehicle_count, current_detections, classIDs, distances, safe, boxs = count_vehicles(idxs, boxes, classIDs,
                                                                                        vehicle_count,
                                                                                        previous_frame_detections,
                                                                                        frame, LABELS, list_of_vehicles)

    # Display Vehicle Count if a vehicle has passed the line 
    displayVehicleCount(frame, vehicle_count)

    # Create a Dataframe with all returned Values 
    dfree = pd.DataFrame(columns=["vehicle_id", "vehicle_class", "accuracy", "speed", "position", "state"])
    dfree["vehicle_id"] = [i[0] for i in idxs.tolist()]
    #dfree["vehicle_id"] = idxs.tolist()
    dfree["vehicle_class"] = classIDsS
    dfree["accuracy"] = confidencesS
    #dfree["speed"] = boxs
    dfree["speed"] = str(boxs)
    dfree["position"] = speeds
    dfree["state"] = safe
    dfree["time"] = now

    print(boxs)

    # Create connection to postgres database
    connect = db.connect('../db.sqlite3')
    # connect = "postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/postgres_db"
    # engine = create_engine(connect)
    engine = create_engine('sqlite:///../db.sqlite3', echo=False)

    # write dataframe to database
    # dfree.to_sql('api_vehicle', con=engine, index=False, if_exists='append')
    # dfree.to_sql('cars_vehicle', con=engine, index=False, if_exists='append')
    # dfree.to_sql('cars_vehicle', con=engine, index=False, if_exists='append')
    dfree.to_sql('cars_vehicle', con=engine, index=False, if_exists='append')

    output = dfree.itertuples()
    dfree.to_sql('cars_vehicle', con=engine, index=False, if_exists='append')
    data = tuple(output)

    # print(data)

    if writer is not None:
        writer.write(frame)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    totalFrames += 1
    fps.update()

# fps.stop()
# print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

if writer is not None:
    writer.release()

if not args.get("input", False):
    vs.stop()

else:
    vs.release()

cv2.destroyAllWindows()
