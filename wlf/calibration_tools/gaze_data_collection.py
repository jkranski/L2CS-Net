import cv2
import numpy as np
import time
from pathlib import Path
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
from utils import select_device, draw_gaze

from face_detection import RetinaFace
from model import L2CS
from PIL import Image
from wlf.utility_scripts.calibration_data import Point2D, BoundingBox


# NOTE: Consider adding Linear layer for u/v, with 4x bins per layer (similar to 90 bins for yaw, gaze)
# NOTE: Also consider looking into applying confidence threshold on gaze detection
def extract_face(full_img, bbox):
    x_min = int(bbox[0])
    if x_min < 0:
        x_min = 0
    y_min = int(bbox[1])
    if y_min < 0:
        y_min = 0
    x_max = int(bbox[2])
    y_max = int(bbox[3])
    bbox_width = x_max - x_min
    bbox_center_x = int((x_max + x_min) / 2.)
    bbox_center_y = int((y_max + y_min) / 2.)
    bbox_height = y_max - y_min
    bbox = BoundingBox(center=Point2D(bbox_center_x, bbox_center_y),
                       upper_left=Point2D(x_min, y_min),
                       lower_right=Point2D(x_max, y_max),
                       width=bbox_width,
                       height=bbox_height)

    # Crop image
    face_img = full_img[y_min:y_max, x_min:x_max]
    return face_img, bbox


def annotate_frame(frame, bounding_box, yaw, pitch, fps):
    # Draw gaze, face bounding box and annotate with FPS
    draw_gaze(bounding_box.upper_left.x, bounding_box.upper_left.y, bounding_box.width, bounding_box.height, frame,
              (yaw, pitch), color=(0, 0, 255))
    cv2.rectangle(frame,
                  (bounding_box.upper_left.x, bounding_box.upper_left.y),
                  (bounding_box.lower_right.x, bounding_box.lower_right.y),
                  (0, 255, 0), 1)
    cv2.putText(frame, f'FPS: {fps:.2f} Pitch (deg): {pitch*180./np.pi:.2f} Yaw (deg): {yaw*180./np.pi:.2f}', (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
    return frame


class GazeDataCollection:
    def __init__(self):
        cam = 0
        self.__init_model__()
        self.__init_camera__(cam)
        self.__init_visualization__()
        # gaze_data: (bbox_center_x, bbox_center_y, bbox_width, bbox_height,
        #             yaw, pitch,
        #             label_x, label_y,
        #             label_u, label_v)
        # Label_x/y: 0 -> 3, for row, column _u/v for pixel value
        self.gaze_data = np.zeros((300, 10), dtype=np.float32)
        self.capture_data = False
        self.capture_idx = 0
        self.label = Point2D(x=0, y=0)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        calibration_data_path = os.path.join(os.getcwd(), 'calibration_data')
        Path(f"./{calibration_data_path}").mkdir(exist_ok=True)
        self.session_path = os.path.join(calibration_data_path, timestr)
        Path(f"./{self.session_path}").mkdir(exist_ok=True)
        self.capture_complete = False

    def __init_model__(self):
        cudnn.enabled = True
        arch = 'ResNet50'
        batch_size = 1
        self.gpu = select_device("0", batch_size=batch_size)
        snapshot_path = "../../models/L2CSNet_gaze360.pkl"

        self.transformations = transforms.Compose([
            transforms.Resize(448),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 90)
        print('Loading snapshot.')
        saved_state_dict = torch.load(
            snapshot_path, map_location=self.gpu)
        self.model.load_state_dict(saved_state_dict)
        self.model.to(self.gpu)
        self.model.eval()
        self.softmax = nn.Softmax(dim=1)
        self.detector = RetinaFace(gpu_id=0)
        self.idx_tensor = [idx for idx in range(90)]
        self.idx_tensor = torch.FloatTensor(self.idx_tensor).to(self.gpu)

    def __init_camera__(self, cam_id):
        self.cam = cv2.VideoCapture(cam_id)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Check if the webcam is opened correctly
        if not self.cam.isOpened():
            raise IOError("Cannot open webcam")

    def __init_visualization__(self):
        # Set up gaze targets
        self.gaze_target = "Gaze Targets"
        cv2.namedWindow(self.gaze_target, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.gaze_target, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        x, y, w, h = cv2.getWindowImageRect(self.gaze_target)
        self.window_x = x
        self.window_y = y
        self.window_w = w
        self.window_h = h
        self.canvas_img = np.zeros((h, w, 3))

    def detect_faces(self, input_img):
        return self.detector(input_img)

    def convert_and_load_img(self, input_img):
        input_img = cv2.resize(input_img, (224, 224))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(input_img)
        input_img = self.transformations(im_pil)
        input_img = Variable(input_img).to(self.gpu)
        output_img = input_img.unsqueeze(0)
        return output_img

    def get_gaze_estimate(self, face_patch):
        yaw, pitch = self.model(face_patch)

        pitch_pred = self.softmax(pitch)
        yaw_pred = self.softmax(yaw)
        # Get continuous predictions in degrees.
        pitch_pred = torch.sum(
            pitch_pred.data[0] * self.idx_tensor) * 4 - 180
        yaw_pred = torch.sum(
            yaw_pred.data[0] * self.idx_tensor) * 4 - 180
        # Detach and get usable values
        pitch_pred = pitch_pred.cpu().detach().numpy() * np.pi / 180.0
        yaw_pred = yaw_pred.cpu().detach().numpy() * np.pi / 180.0

        return pitch_pred, yaw_pred

    def run(self):
        with torch.no_grad():
            while True:
                success, frame = self.cam.read()
                start_fps = time.time()
                frame = cv2.flip(frame, 1)

                # Draw target images on canvas
                i = self.label.x
                j = self.label.y
                width_pos = int(self.window_w / (2 * 4) + i * self.window_w / 4)
                height_pos = int(self.window_h / (2 * 4) + j * self.window_h / 4)
                circle_color = (0, 0, 255) if self.capture_data else (0, 255, 255)
                cv2.circle(self.canvas_img, (width_pos, height_pos), 10, circle_color, -1)

                faces = self.detect_faces(frame)
                if faces is not None:
                    for box, landmarks, score in faces:
                        if score < .95:
                            continue
                        face, bbox = extract_face(frame, box)

                        converted_face = self.convert_and_load_img(face)
                        pitch_predicted, yaw_predicted = self.get_gaze_estimate(converted_face)
                        my_fps = 1.0 / (time.time() - start_fps)
                        frame = annotate_frame(frame, bbox, yaw_predicted, pitch_predicted, my_fps)
                        if self.capture_data:
                            # gaze_data: (bbox_center_x, bbox_center_y, bbox_width, bbox_height,
                            #             yaw, pitch,
                            #             label_x, label_y,
                            #             label_u, label_v)
                            self.gaze_data[self.capture_idx] = [bbox.center.x, bbox.center.y,
                                                                bbox.width, bbox.height,
                                                                yaw_predicted, pitch_predicted,
                                                                self.label.x, self.label.y,
                                                                width_pos, height_pos]
                            self.capture_idx += 1
                            print(f"capture_count: {self.capture_idx} gaze_data: {self.gaze_data.shape[0]}")
                            if self.capture_idx == self.gaze_data.shape[0]:
                                self.capture_data = False
                                self.capture_idx = 0
                                with open(f"{self.session_path}/target_{self.label.x}_{self.label.y}.npy", 'wb') as f:
                                    np.save(f, self.gaze_data)
                                self.canvas_img[:, :, :] = 0
                                if self.label.x == 3:
                                    self.label.x = 0
                                    self.label.y += 1
                                else:
                                    self.label.x += 1
                                if self.label.y == 4:
                                    print(f"Reached end of data capture")
                                    self.capture_complete = True
                                    break
                cv2.imshow("Demo", frame)
                cv2.imshow(self.gaze_target, self.canvas_img)
                key = cv2.waitKey(1)

                if key & 0xFF == 27 or self.capture_complete:
                    # ESC pressed
                    print("Closing...")
                    break
                elif key & 0xFF == 32:  # Spacebar
                    print(f"Capturing gaze data for location x: {self.label.x} y: {self.label.y}")
                    self.capture_data = True
                elif key != -1:
                    print(f"Key pressed: {key}")

            self.cam.release()
            cv2.destroyAllWindows()


app = GazeDataCollection()
app.run()

# # ~30 FPS. Want around 1000 samples per position, run for ~30 sec
# # Pre-allocate a numpy array with a label column of size 1000 and collect data until it's filled
