import cv2
import mediapipe as mp
import time
import argparse
import math
import numpy as np
import os
import csv

class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)


    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        lmList = []  # Initialize lmList here
        if self.results.pose_landmarks and draw:
            for lm_id, lm in enumerate(self.results.pose_landmarks.landmark):
                if lm_id == 0 or lm_id > 10:
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([lm_id, cx, cy])
            lmList = sorted(lmList, key=lambda x: x[0])

            connections = self.mpPose.POSE_CONNECTIONS
            for connection in connections:
                startingindex, endingindex = connection
                if startingindex > 10 and endingindex > 10:
                    start_point = tuple(lmList[startingindex-11][1:])
                    end_point = tuple(lmList[endingindex-11][1:])
                    cv2.line(img, start_point, end_point, (255, 255, 255), 2)
            for lm in lmList:
                cv2.circle(img, (lm[1], lm[2]), 3, (0, 0, 255), cv2.FILLED)

        return lmList

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                if id == 0 or id > 10:
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
            lmList = sorted(lmList, key=lambda x: x[0])

            if draw:
                # Draw a circle on joint 14 (left wrist) for demonstration purposes
                cv2.circle(img, (lmList[14][1], lmList[14][2]), 3, (255, 0, 0), cv2.FILLED)

        return lmList

    def calculateDistances(self, lmList):
        distances = []
        num_joints = len(lmList)
        for i in range(num_joints):
            for j in range(i+1, num_joints):
                joint_1 = lmList[i]
                joint_2 = lmList[j]
                dist = math.sqrt((joint_2[1] - joint_1[1])**2 + (joint_2[2] - joint_1[2])**2)
                distances.append([joint_1[0], joint_2[0], int(dist)])
        return distances

# def write_to_csv(data):
#     csv_file = 'output.csv'

#     # Writing the 2D list to a CSV file
#     with open(csv_file, 'w', newline='') as csvfile:
#         csv_writer = csv.writer(csvfile)

#         # Iterate through the 2D list and write each row to the CSV file
#         for row in data:
#             csv_writer.writerow(row)

def count_files(folder_path):
    try:
        # List all files in the folder
        files = os.listdir(folder_path)

        # Count the number of files in the folder
        num_files = len(files)

        return num_files

        # If you want to list the file names, you can uncomment the following line
        # print('File names:', files)
    except FileNotFoundError:
        print(f'The folder "{folder_path}" was not found.')
    except PermissionError:
        print(f'Permission error: Unable to access "{folder_path}".')

def process_image(source):
    img = cv2.imread(source)
    if img is None:
        print(f"Failed to open {source}")
        return

    detector = poseDetector(detectionCon=True, trackCon=True)
    detector.findPose(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
            # print("Landmarks: ")

            # for lm in lmList:
            #     print(f"Joint {lm[0]}: ({lm[1]}, {lm[2]})")

            distances = detector.calculateDistances(lmList)
            # print("Distances:")

            # for distance in distances:
            #     joint1, joint2, dist = distance
            #     print(f"Distance between Joint {joint1} and Joint {joint2}: {dist:.2f}")
            # print(type(distances))
            # for distance in distances:
            #     print(distance[2],end=" ")
            # DistancesList=[i[2] for i in distances]
            # return DistancesList
            points=[[i[1],i[2]] for i in lmList]
            return points
    else: print('error at '+source)
    

def frame(source_loc):
    parser = argparse.ArgumentParser(description='Pose Estimation')
    parser.add_argument('--source', type=str, default=source_loc, help='Path to video file or image file or "webcam" for webcam feed')
    parser.add_argument('--mode', type=str, default='image', help='Mode: "video", "webcam", or "image"')
    args = parser.parse_args()
    return process_image(args.source)

def sequence(seqNo): #to call each frame in a given sequence
    seqData=[]

    if seqNo<10:
        n='0'+str(seqNo)
    else:
        n=str(seqNo)
    location='datasets/fall-'+n+'-cam0-rgb'
    framecount=count_files(location)
    print(seqNo,framecount)
    for i in range(1,framecount+1):
        if i<100:
            x='0'*(3-len(str(i)))+str(i)
        else:
            x=str(i)
        print(seqNo,x)
        frameLocation='datasets/fall-'+n+'-cam0-rgb/fall-'+n+'-cam0-rgb-'+str(x)+'.png'
        frameData=frame(frameLocation)
        if frameData==None:
            continue
        seqData+=frameData
    # seqData=seqData+frame(location+"/fall-01-cam0-rgb-001.png")
    return seqData
def dataset(seqCount): #to call each sequence
    # i=seqCount
    for i in range(seqCount,seqCount+1):
        data=sequence(i)
        file=open('xydat.txt','a')
        for j in range(len(data)):
            
            file.write(str(data[j][0])+' '+str(data[j][1]))
            if j!=len(data)-1:
                file.write(' ')  
        file.write('\n')  
        file.close()
        print('Seq',i,'saved')
        data.clear()
    # return data

# data=dataset(30)
# # write_to_csv(data)
# print('Rows :',len(data))
# print('Columns :')
# for i in data:
#     print(len(i))

dataset(30)