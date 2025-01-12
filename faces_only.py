import argparse
import numpy as np
import cv2
import time
import mido


import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from utils import select_device, draw_gaze
from PIL import Image, ImageOps

from face_detection import RetinaFace
from model import L2CS

from wlf import GazeData, Face, Point2DF, Point3DF, GazeSender
import redis
import io
import base64


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--cam', dest='cam_id', help='Camera device id to use [0]',
        default=0, type=int)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    batch_size = 1
    cam = args.cam_id
    gpu = select_device(args.gpu_id, batch_size=batch_size)

    cap = cv2.VideoCapture(cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    detector = RetinaFace(device=gpu)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    x = 0

    redis_client = redis.Redis()
    sender = GazeSender(redis_client)
    with torch.no_grad():
        while True:
            success, frame = cap.read()
            start_fps = time.time()
            frame = cv2.flip(frame, 1)

            faces = detector(frame)
            if faces is not None:
                net_faces: list[Face] = []
                for box, landmarks, score in faces:
                    if score < .95:
                        continue
                    x_min = int(box[0])
                    if x_min < 0:
                        x_min = 0
                    y_min = int(box[1])
                    if y_min < 0:
                        y_min = 0
                    x_max = int(box[2])
                    y_max = int(box[3])
                    bbox_width = x_max - x_min
                    bbox_center = (x_max + x_min)/2.
                    bbox_center_y = (y_max + y_min) / 2.
                    bbox_height = y_max - y_min\

                    # Crop image
                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.resize(img, (224, 224))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    
                    cv2.rectangle(frame, (x_min, y_min),
                                  (x_max, y_max), (0, 255, 0), 1)

                    jpeg_img = io.BytesIO()
                    im_pil.save(jpeg_img, format='JPEG')
                    net_face = Face(camera_centroid_norm=Point2DF(x=float(bbox_center)/frame.shape[1], y=float(bbox_center_y)/frame.shape[0]),
                                    gaze_vector=Point3DF(x=0.0, y=0.0, z=0.0),
                                    gaze_screen_intersection_norm=Point2DF(
                                        x=0, y=0.0),
                                    face_patch_jpeg_base64=base64.b64encode(
                        jpeg_img.getvalue())
                    )
                    net_faces.append(net_face)

                sender.send(GazeData(faces=net_faces))
            
            myFPS = 1.0 / (time.time() - start_fps)
            cv2.putText(frame, 'FPS: {:.1f}'.format(
                myFPS), (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Demo", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
