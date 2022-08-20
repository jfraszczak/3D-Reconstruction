import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils

class MarkersMaskExtractor:
    def __init__(self, img, angle):
        self.img_original = img
        self.angle = angle
        self.img_hsv = None

        self.green_mask = None
        self.black_mask = None
        self.markers_mask = None

    def preprocessImage(self):
        img_hsv = cv2.cvtColor(self.img_original, cv2.COLOR_RGB2HSV)
        self.img_hsv = cv2.GaussianBlur(img_hsv, (5, 5), 0)

    def findGreenMask(self):
        lower_bound = np.array([50, 10, 10])
        upper_bound = np.array([90, 255, 255])
        mask = cv2.inRange(self.img_hsv, lower_bound, upper_bound)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        self.green_mask = mask

    def findBlackMask(self):
        lower_bound = np.array([0, 0, 0])
        upper_bound = np.array([360, 255, 40])
        mask = cv2.inRange(self.img_hsv, lower_bound, upper_bound)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        self.black_mask = mask

    def applyMaskAroundPerson(self):
        y, x = self.img_original.shape[0], self.img_original.shape[1]
        mask = np.zeros((y, x), dtype=np.uint8)
        y_start, y_stop = int(0.2 * y), int(0.67 * y)

        if self.angle is None:
            x_start, x_stop = int(0.1 * x), int(0.9 * x)
        elif self.angle < 180:
            x_start, x_stop = int(0.1 * x), int(0.8 * x)
        elif self.angle > 180:
            x_start, x_stop = int(0.2 * x), int(0.9 * x)
        elif self.angle == 0 or self.angle == 180:
            x_start, x_stop = int(0.3 * x), int(0.7 * x)

        mask[y_start:y_stop, x_start:x_stop] = 255 * np.ones((y_stop - y_start, x_stop - x_start), dtype=np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)

        self.green_mask = cv2.bitwise_and(self.green_mask, self.green_mask, mask=mask)
        self.black_mask = cv2.bitwise_and(self.black_mask, self.black_mask, mask=mask)

    def createMarkersMask(self):
        markers_mask = self.green_mask + self.black_mask
        kernel = np.ones((5, 5), np.uint8)
        markers_mask = cv2.erode(markers_mask, kernel, iterations=1)
        markers_mask = cv2.dilate(markers_mask, kernel, iterations=4)
        markers_mask = cv2.erode(markers_mask, kernel, iterations=3)
        self.markers_mask = markers_mask

    def visualize(self):
        extracted_markers = cv2.bitwise_and(self.img_original, self.img_original, mask=self.markers_mask)

        plt.imshow(extracted_markers[:, :, ::-1])
        plt.savefig(utils.dir_results + '/markers' + str(self.angle) + '.png')
        plt.clf()

        plt.imshow(self.markers_mask, cmap='gray')
        plt.savefig(utils.dir_results + '/markers_mask' + str(self.angle) + '.png')
        plt.clf()

    def extract_markers_masks(self, visualize=False):
        self.preprocessImage()
        self.findGreenMask()
        self.findBlackMask()
        self.applyMaskAroundPerson()
        self.createMarkersMask()

        if visualize:
            self.visualize()

    def getMask(self):
        return self.markers_mask
