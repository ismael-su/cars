import math
import os
import time

import cv2
import numpy as np
from scipy import spatial

FRAMES_BEFORE_CURRENT = 10


# function to calculate the speed of each vehicle
def estimateSpeed(location1, location2, location3, location4):
    d_pixels1 = math.sqrt((location2[0] - location1[0]) ** 2 + (location2[1] - location1[1]) ** 2)
    d_pixels2 = math.sqrt((location4[0] - location3[0]) ** 2 + (location4[1] - location3[1]) ** 2)
    d_pixels = (d_pixels1 + d_pixels2) / 2

    carWidth = 2
    ppm1 = location2[1] / int(carWidth)
    ppm2 = location4[1] / int(carWidth)
    ppm = (ppm1 + ppm2) / 2
    # print(location2[1], d_pixels)
    # ppm = 8.8
    d_meters = d_pixels / ppm
    # print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
    fps = 500
    speed = d_meters * fps * 3.6
    return speed


# PURPOSE: Displays the vehicle count on the top-left corner of the frame
# PARAMETERS: Frame on which the count is displayed, the count number of vehicles 
# RETURN: N/A
def displayVehicleCount(frame, vehicle_count):
    cv2.putText(
        frame,  # Image
        'Detected Vehicles: ' + str(vehicle_count),  # Label
        (30, 30),  # Position
        cv2.FONT_HERSHEY_SIMPLEX,  # Font
        0.8,  # Size
        (0, 0, 0),  # Color
        2,  # Thickness
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
    )


# PURPOSE: Displaying the FPS of the detected video
# PARAMETERS: Start time of the frame, number of frames within the same second
# RETURN: New start time, new number of frames 
def displayFPS(start_time, num_frames):
    current_time = int(time.time())
    if (current_time > start_time):
        os.system('clear')  # Equivalent of CTRL+L on the terminal
        print("FPS:", num_frames)
        num_frames = 0
        start_time = current_time
    return start_time, num_frames


# PURPOSE: Draw all the detection boxes with a green dot at the center
# RETURN: N/A
def drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame, COLORS, LABELS):
    # ensure at least one detection exists
    classIDsS = []
    confidencesS = []
    speeds = []
    if len(idxs) > 0:
        # loop over the indices we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            classIDsS.append(LABELS[classIDs[i]])
            confidencesS.append(confidences[i])
            # Draw a green dot in the middle of the box
            cv2.circle(frame, (x + (w // 2), y + (h // 2)), 2, (0, 0xFF, 0), thickness=2)
            # , speeds
    return classIDsS, confidencesS, 1


# PURPOSE: Identifying if the current box was present in the previous frames
# PARAMETERS: All the vehicular detections of the previous frames, 
# the coordinates of the box of previous detections
# RETURN: True if the box was current box was present in the previous frames;
# False if the box was not present in the previous frames

def boxInPreviousFrames(previous_frame_detections, current_box, current_detections, FRAMES_BEFORE_CURRENT):
    centerX, centerY, width, height = current_box
    dist = np.inf  # Initializing the minimum distance
    # Iterating through all the k-dimensional trees
    for i in range(FRAMES_BEFORE_CURRENT):
        coordinate_list = list(previous_frame_detections[i].keys())
        if len(coordinate_list) == 0:  # When there are no detections in the previous frame
            continue
        # Finding the distance to the closest point and the index
        temp_dist, index = spatial.KDTree(coordinate_list).query([(centerX, centerY)])
        if (temp_dist < dist):
            dist = temp_dist
            frame_num = i
            coord = coordinate_list[index[0]]

    if (dist > (max(width, height) / 2)):
        return False

    # Keeping the vehicle ID constant
    current_detections[(centerX, centerY)] = previous_frame_detections[frame_num][coord]
    return True


# function to count, detect, track vahicles and calculate speed, distance between objects 
def count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame, LABELS, list_of_vehicles):
    current_detections = {}
    distances = []
    safes = []
    boxs = []
    distance = []
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indices we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            boxs.append((x, y, w, h))
            centerX = x + (w // 2)
            centerY = y + (h // 2)
            # When the detection is in the list of vehicles, AND
            # it crosses the line AND
            # the ID of the detection is not present in the vehicles
            if (LABELS[classIDs[i]] in list_of_vehicles):
                current_detections[(centerX, centerY)] = vehicle_count
            if (not boxInPreviousFrames(previous_frame_detections, (centerX, centerY, w, h), current_detections,
                                        FRAMES_BEFORE_CURRENT)):
                vehicle_count += 1

            # Add the current detection mid-point of box to the list of detected items
            # Get the ID corresponding to the current detection
            ID = current_detections.get((centerX, centerY))
            # If there are two detections having the same ID due to being too close, 
            # then assign a new ID to current detection.
            if (list(current_detections.values()).count(ID) > 1):
                current_detections[(centerX, centerY)] = vehicle_count
                vehicle_count += 1

            for b in range(len(boxes)):
                for k in range(b + 1, len(boxes)):
                    di = distObj(boxes[b][0], boxes[b][1], boxes[b][2], boxes[b][3], boxes[k][0], boxes[k][1],
                                 boxes[k][2], boxes[k][3])
                    distances.append(di)
            if len(idxs) > 0:
                status = []
                idf = idxs.flatten()
                close_pair = []
                S_close_pair = []
                center = []
                co_info = []
                for i in idf:
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    cen = [int(x + w / 2), int(y + h / 2)]

                    center.append(cen)
                    cv2.circle(frame, tuple(cen), 1, (0, 0, 0), 1)
                    co_info.append([w, h, cen])
                    status.append(0)
                for i in range(len(center)):
                    for j in range(len(center)):
                        g = isclose(co_info[i], co_info[j])
                        if g == 1:
                            close_pair.append([center[i], center[j]])
                            centerS, Dists = DistCenter(center[i][0], center[i][1], center[j][0], center[j][1])
                            status[i] = 1
                            status[j] = 1
                        elif g == 2:
                            S_close_pair.append([center[i], center[j]])
                            if status[i] != 1:
                                status[i] = 2
                            if status[j] != 1:
                                status[j] = 2
                        else:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 120, 255), 2)

                total_p = len(center)
                low_risk_p = status.count(2)
                high_risk_p = status.count(1)
                safe_p = status.count(0)
                kk = 0
                for i in idf:
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    cens = [int(x + w / 2), int(y + h / 2)]
                    if status[kk] == 1:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)
                        if distances[i] < 35:
                            dis = "{:.2f} m".format(distances[i])
                            cv2.putText(frame, "dist: " + dis, (cens[0], cens[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (250, 0, 0), 2)
                    elif status[kk] == 0:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 120, 255), 2)
                        if distances[i] < 35:
                            dis = "{:.2f} m".format(distances[i])
                            cv2.putText(frame, "dist: " + dis, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 0, 0), 2)

                    kk += 1

                for h in close_pair:
                    cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)
                for b in S_close_pair:
                    cv2.line(frame, tuple(b[0]), tuple(b[1]), (0, 255, 255), 2)

            # Notifications about the safety of the each vehicle
            if di < 10:
                safe = "distance very small"
                safes.append(safe)
            elif di < 20:
                safe = "Good distance"
                safes.append(safe)
            else:
                safe = "SAFE"
                safes.append(safe)

            # Display the ID at the center of the box
            cv2.putText(frame, str(ID), (centerX + 3, centerY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)

    return vehicle_count, current_detections, classIDs, distances, safes, boxs


def midpoint(x, y, w, h):
    return (x + w) * 0.5, (y + h) * 0.5


def distObj(x1, y1, w1, h1, x2, y2, w2, h2):
    cx1, cy1 = midpoint(x1, y1, w1, h1)
    cx2, cy2 = midpoint(x2, y2, w2, h2)
    distance = math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)
    return distance


def DistCenter(x1, y1, x2, y2):
    center = []
    dists = []
    centerX, centerY = ((x1 + x2) / 2), ((y1 + y2) / 2)
    center.append([centerX, centerY])
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    dists.append(distance)
    return center, dists


# # speed estimation
# def estimateSpeed(location1, location2, ppm, fs):
# 	d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
# 	d_meters = d_pixels/ppm
# 	speed = d_meters*fs*3.6
# 	return speed

def midPoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def dist(center1, center2):
    # return math.sqrt((x2-x1)**2 + (y2-y1) ** 2)
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def rect_distance(x1, y1, x1b, y1b, x2, y2, x2b, y2b):
    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    if top and left:
        return dist(x1, y1b, x2b, y2)
    elif left and bottom:
        return dist((x1, y1), (x2b, y2b))
    elif bottom and right:
        return dist((x1b, y1), (x2, y2b))
    elif right and top:
        return dist((x1b, y1b), (x2, y2))
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:  # rectangles intersect
        return 0.


def rectangleCenter(x, y, w, h):
    return (x + (w // 2), y + (h // 2))


# Calibration needed for each video
def calibrated_dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + 550 / ((p1[1] + p2[1]) / 2) * (p1[1] - p2[1]) ** 2) ** 0.5


def isclose(p1, p2):
    c_d = calibrated_dist(p1, p2)
    calib = (p1[1] + p2[1]) / 2
    if 0 < c_d < 0.15 * calib:
        return 1
    elif 0 < c_d < 0.2 * calib:
        return 2
    else:
        return 0
