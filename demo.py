import argparse
import os
import joblib

import numpy as np
import cv2
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from utils import select_device, draw_gaze, annotate_image_debug
from PIL import Image, ImageOps

from face_detection import RetinaFace
from model import L2CS

from wlf import GazeData, Face, Point2DF, Point3DF, GazeSender
from wlf.calibration_tools.regression_nn import RegressionNeuralNetwork, ClassificationNeuralNetwork
import redis
import io
import base64

#TODO: Cleanup old code.

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)
    parser.add_argument(
        '--training_timestr', dest='training_timestr', help='Timestring for training time of projection mapping model '
                                                            'and scalar',
        default="20240131-202532", type=str)
    parser.add_argument(
        '--data_timestr', dest='data_timestr', help='Timestring for data collection',
        default="20240131", type=str)
    parser.add_argument(
        '--cam', dest='cam_id', help='Camera device id to use [0]',
        default=1, type=int)
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)
    parser.add_argument(
        '--classification', dest='classification', default=False, action='store_true')

    args = parser.parse_args()
    return args


def getArch(arch, bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS(torchvision.models.resnet.BasicBlock, [2, 2,  2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS(torchvision.models.resnet.BasicBlock, [3, 4,  6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                  'The default value of ResNet50 will be used instead!')
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], bins)
    return model


if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    arch = args.arch
    batch_size = 1
    cam = args.cam_id
    gpu = select_device(args.gpu_id, batch_size=batch_size)
    snapshot_path = args.snapshot
    classification = args.classification

    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    model = getArch(arch, 90)
    print('Loading snapshot.')
    saved_state_dict = torch.load(
        snapshot_path, map_location=gpu)
    model.load_state_dict(saved_state_dict)
    model.to(gpu)
    model.eval()

    #proj_model_timestr = args.projection_timestr

    projection_model = ClassificationNeuralNetwork().to(gpu) if classification else RegressionNeuralNetwork().to(gpu)
    model_timestr = f"{args.training_timestr}_{args.data_timestr}"
    projection_model.load_state_dict(torch.load(os.path.join(os.getcwd(), f"wlf\\calibration_tools\\calibration_models\\{model_timestr}_model.ckpt")))
    projection_model.eval()
    input_scalar = joblib.load(f"wlf\\calibration_tools\\calibration_models\\{model_timestr}_scalar.bin")

    softmax = nn.Softmax(dim=1)
    detector = RetinaFace(gpu_id=0)
    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(gpu)
    x = 0

    cap = cv2.VideoCapture(cam)
    # Set resolution
    print("Frame default resolution: (" + str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + "; " + str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + ")")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    print("Frame resolution set to: (" + str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + "; " + str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + ")")


    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    redis_client = redis.Redis()
    sender = GazeSender(redis_client)

    #Set up OpenCV Window
    cv2.namedWindow("Demo")

    with torch.no_grad():
        while True:
            success, frame = cap.read()
            start_fps = time.time()
            frame = cv2.flip(frame, 1)
            faces = detector(frame)
            quadrants = [0, 0, 0, 0, 0]
            gaze_draw_data = []
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
                    bbox_center_x = (x_max + x_min) / 2.
                    bbox_center_y = (y_max + y_min) / 2.
                    bbox_height = y_max - y_min

                    # Crop image
                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.resize(img, (224, 224))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    img = transformations(im_pil)
                    img = Variable(img).to(gpu)
                    img = img.unsqueeze(0)

                    # gaze prediction
                    gaze_yaw, gaze_pitch = model(img)

                    pitch_predicted = softmax(gaze_pitch)
                    yaw_predicted = softmax(gaze_yaw)

                    # Get continuous predictions in degrees.
                    pitch_predicted = torch.sum(
                        pitch_predicted.data[0] * idx_tensor) * 4 - 180
                    yaw_predicted = torch.sum(
                        yaw_predicted.data[0] * idx_tensor) * 4 - 180

                    pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi/180.0
                    yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi/180.0

                    gaze_draw_data.append([x_min, y_min, bbox_width, bbox_height, yaw_predicted, pitch_predicted])

                    # 1920/1280 to account for training at 1920x1080 and running at 1280x720
                    scale_factor = 1920./1280.
                    # scale_factor = 1.
                    projection_input = torch.tensor([scale_factor*bbox_center_x, scale_factor*bbox_center_y,
                                                     scale_factor*bbox_width, scale_factor*bbox_height,
                                                     yaw_predicted, pitch_predicted],
                                                    dtype=torch.float32)
                    projection_input = input_scalar.transform(projection_input.reshape(1, -1))
                    projection_input = torch.tensor(projection_input, dtype=torch.float32)
                    y_pred = projection_model(projection_input.to(gpu))
                    if classification:
                        y_pred_probs = torch.softmax(y_pred, dim=1)
                        y_pred = torch.argmax(y_pred, 1)

                    y_pred = y_pred.cpu().detach().numpy()

                    myFPS = 1.0 / (time.time() - start_fps)
                    print(f"FPS: {myFPS}")
                    if classification:
                        responses = ["A", "B", "C", "D"]
                        centroids = [0.125, 0.375, 0.625, 0.875]
                        # Screen not flipped during training
                        response = responses[int(y_pred[0])]
                        centroid = centroids[int(y_pred[0])]
                        # response = responses[3 - int(y_pred[0])]
                        # centroid = centroids[3 - int(y_pred[0])]

                    else:
                        scaled_output = (y_pred[0, 0]-500)/2500.
                        response = "None"
                        if scaled_output < 0.15:
                            response = "A"
                        elif 0.25 < scaled_output < 0.45:
                            response = "B"
                        elif 0.55 < scaled_output < 0.75:
                            response = "C"
                        elif scaled_output > 0.85:
                            response = "D"

                    jpeg_img = io.BytesIO()
                    im_pil.save(jpeg_img, format='JPEG')
                    net_face = Face(camera_centroid_norm=Point2DF(x=float(bbox_center_x)/frame.shape[1],
                                                                  y=float(bbox_center_y)/frame.shape[0]),
                                    gaze_vector=Point3DF(x=0.0, y=0.0, z=0.0),
                                    gaze_screen_intersection_norm=Point2DF(
                                        x=centroid, y=0.0),
                                    face_patch_jpeg_base64=base64.b64encode(
                                        jpeg_img.getvalue())
                                    )
                    # Lowered from 0.75 due to limited training set. FIXME
                    if y_pred_probs.max(dim=1).values > 0.20:
                        net_faces.append(net_face)
                        print(f"response: {response}")
                    # print(net_face)

                for x_min, y_min, bbox_width, bbox_height, yaw_predicted, pitch_predicted in gaze_draw_data:
                    draw_gaze(x_min, y_min, bbox_width, bbox_height, frame,
                              (yaw_predicted, pitch_predicted), color=(0, 0, 255))
                sender.send(GazeData(faces=net_faces))

            cv2.imshow("Demo", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
