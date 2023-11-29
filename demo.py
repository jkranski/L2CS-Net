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
        '--cam', dest='cam_id', help='Camera device id to use [0]',
        default=0, type=int)
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

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

    softmax = nn.Softmax(dim=1)
    detector = RetinaFace(gpu_id=0)
    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(gpu)
    x = 0

    cap = cv2.VideoCapture(cam)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Set up MIDI port
    msg = mido.Message('note_on', note=60)
    out_port = mido.open_output(mido.get_output_names()[0])
    msg_changed = False

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
                    bbox_height = y_max - y_min
                    # x_min = max(0,x_min-int(0.2*bbox_height))
                    # y_min = max(0,y_min-int(0.2*bbox_width))
                    # x_max = x_max+int(0.2*bbox_height)
                    # y_max = y_max+int(0.2*bbox_width)
                    # bbox_width = x_max - x_min
                    # bbox_height = y_max - y_min

                    # Crop image
                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.resize(img, (224, 224))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    img = transformations(im_pil)
                    img = Variable(img).to(gpu)
                    img = img.unsqueeze(0)

                    # gaze prediction
                    gaze_pitch, gaze_yaw = model(img)

                    pitch_predicted = softmax(gaze_pitch)
                    yaw_predicted = softmax(gaze_yaw)

                    # Get continuous predictions in degrees.
                    pitch_predicted = torch.sum(
                        pitch_predicted.data[0] * idx_tensor) * 4 - 180
                    yaw_predicted = torch.sum(
                        yaw_predicted.data[0] * idx_tensor) * 4 - 180

                    pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi/180.0
                    yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi/180.0

                    # Check for gaze direction
                    # Seems to have pitch and yaw swapped. Pitch here is the left/right gaze of the viewer
                    left_right_gaze = pitch_predicted*180.0/np.pi
                    # Determine pierce point
                    camera_yaw = -1*pitch_predicted  # right of camera is +, left of camera is -
                    camera_stage_distance = 0.6097  # ~24 in
                    # TODO: Implement lookup from yaml or realtime tool
                    stage_plane_x = 7.674E-4 * (bbox_center - 320)
                    stage_camera_yaw = np.arctan(
                        camera_stage_distance/stage_plane_x) if stage_plane_x != 0. else np.pi/2.
                    stage_gaze_yaw = np.pi - stage_camera_yaw - camera_yaw
                    stage_pillar_distance = 0.6097  # TODO: Should be based on yaml
                    x = stage_plane_x + stage_pillar_distance / \
                        np.tan(stage_gaze_yaw)
                    pierce_point_x = stage_plane_x + \
                        stage_pillar_distance*np.tan(pitch_predicted)
                    # TODO: Add np.bin usage here
                    # Added in some dead bands
                    if left_right_gaze < -20.0:
                        note_tone = 20
                    elif -15. < left_right_gaze < -2.5:
                        note_tone = 40
                    elif 2.5 < left_right_gaze < 15.:
                        note_tone = 60
                    elif left_right_gaze > 20.0:
                        note_tone = 80
                    else:
                        note_tone = msg.note
                    if note_tone != msg.note:
                        msg = msg.copy(note=note_tone)
                        msg_changed = True

                    draw_gaze(x_min, y_min, bbox_width, bbox_height, frame,
                              (pitch_predicted, yaw_predicted), color=(0, 0, 255))
                    cv2.rectangle(frame, (x_min, y_min),
                                  (x_max, y_max), (0, 255, 0), 1)
                    net_face = Face(camera_centroid_norm=Point2DF(x=float(bbox_center)/frame.shape[1], y=float(bbox_center_y)/frame.shape[0]),
                                    gaze_vector=Point3DF(x=0.0, y=0.0, z=0.0),
                                    gaze_screen_intersection_norm=Point2DF(
                                        x=x, y=0.0)
                                    )
                    net_faces.append(net_face)

                sender.send(GazeData(faces=net_faces))
            if msg_changed:
                out_port.send(msg)
                msg_changed = False
            myFPS = 1.0 / (time.time() - start_fps)
            cv2.putText(frame, 'FPS: {:.1f}'.format(
                myFPS), (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Pitch: {:.1f}'.format(-1*pitch_predicted*180./np.pi),
                        (10, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Yaw: {:.1f}'.format(yaw_predicted*180./np.pi), (10, 120),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Pierce Point X: {:.3f}'.format(
                x), (10, 170), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Bbox center: {:.3f}'.format(
                bbox_center), (10, 220), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Stage Plane X: {:.3f}'.format(
                stage_plane_x), (10, 270), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, 'D tan theta: {:.3f}'.format(
                stage_pillar_distance*np.tan(-1*pitch_predicted)), (10, 320), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, 'stage gaze yaw {:.3f}'.format(
                stage_gaze_yaw*180./np.pi), (10, 370), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Demo", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
