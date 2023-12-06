# Based on code from: https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
# Tool to select points of a reference plane in an image to generate a rectified warped image, parallel to stage plane
# Point selection needs to be in order. Top-left, top-right, bottom-right, then bottom-left.
# Points should be selected on a physical plane temporarily held in front of camera, at distance approximately equal
# to stage plane distance. Perhaps use a piece of foam core suspended by a c-clamp stand

# import the necessary packages
import numpy as np
import cv2
from wlf.utility_scripts.calibration_data import PerspectiveTransform
import time


def order_points(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def get_perspective_transform(pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordinates or the top-right and top-left x-coordinates
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    # return the warped image
    return M, max_width, max_height


def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 100, (0, 255, 255), 2)


class GetPerspectiveTransform:
    def __init__(self,
                 cam_id=0):
        self.original_window_name = "Original"
        cv2.namedWindow(self.original_window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.original_window_name, self.on_click)
        self.warped_window_name = "Warped"
        cv2.namedWindow(self.warped_window_name, cv2.WINDOW_AUTOSIZE)
        self.video_stream = cv2.VideoCapture(cam_id)
        ret, self.capture_frame = self.video_stream.read()
        if not ret:
            print("failed to grab frame")
            self.capture_frame = np.zeros((5, 5, 3))
        self.click_locations = []
        self.perspective_transform = None
        cv2.imshow(self.original_window_name, self.capture_frame)
        cv2.imshow(self.warped_window_name, self.capture_frame)
        cv2.waitKey(1)

    def on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_locations.append([x, y])

    def run(self):
        while True:
            # Grab frame
            ret, self.capture_frame = self.video_stream.read()
            if self.perspective_transform is None and len(self.click_locations) >= 4:
                # Compute perspective transform, but only once
                M, w, h = get_perspective_transform(np.array(self.click_locations))
                self.perspective_transform = PerspectiveTransform(matrix=M, width=w, height=h)
            if self.perspective_transform is not None:
                warped_image = cv2.warpPerspective(self.capture_frame,
                                                   self.perspective_transform.matrix,
                                                   (self.perspective_transform.width,
                                                    self.perspective_transform.height))
                cv2.imshow(self.warped_window_name, warped_image)
            for x, y in self.click_locations:
                cv2.circle(self.capture_frame, (x, y), 5, (0, 255, 255), -1)
            cv2.imshow(self.original_window_name, self.capture_frame)
            key = cv2.waitKey(1)
            if key & 0xFF == 27:
                break
            elif key == ord('s'):
                if self.perspective_transform is None:
                    print(f"No perspective transform calculated, select corner points")
                else:
                    current_time = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"{current_time}rectification_data.npy"
                    print(f"Saving data to {filename}")
                    with open(filename, 'wb') as f:
                        np.savez(f,
                                 calibration_matrix=self.perspective_transform.matrix,
                                 output_image_shape=np.array([self.perspective_transform.width,
                                                              self.perspective_transform.height]))

        cv2.destroyAllWindows()
        self.video_stream.release()


if __name__ == '__main__':
    pt = GetPerspectiveTransform(1)
    pt.run()
