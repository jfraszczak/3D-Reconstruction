import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

import MarkersMaskExtractor
import MarkerLinesCollection
import utils


class MarkersCornersExtractor:
    def __init__(self, img, angle):
        self.img = img
        self.angle = angle
        self.img_x = img.shape[1]
        self.img_y = img.shape[0]
        self.mask = None

        self.front_marker = None
        self.back_marker = None
        self.right_marker = None
        self.left_marker = None

    def extractMask(self, visualize=False):
        markers_mask_extractor = MarkersMaskExtractor.MarkersMaskExtractor(self.img, self.angle)
        markers_mask_extractor.extract_markers_masks(visualize=visualize)
        self.mask = markers_mask_extractor.getMask()

    def findLines(self):
        lines = cv2.HoughLinesP(self.mask, 1, np.pi / 180, 10, None, 200, 10)
        img_with_lines = self.img.copy()

        vertical_lines = []
        horizontal_lines = []

        if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                angle = math.atan((l[3] - l[1]) / (l[2] - l[0])) / math.pi * 180

                if 80 < angle < 100 or -100 < angle < -80:
                    cv2.line(img_with_lines, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
                    vertical_lines.append(l)

                if -10 < angle < 10:
                    cv2.line(img_with_lines, (l[0], l[1]), (l[2], l[3]), (255, 0, 255), 3, cv2.LINE_AA)
                    horizontal_lines.append(l)

        plt.imshow(img_with_lines[:, :, ::-1])
        plt.savefig(utils.dir_results + '/markers_lines' + str(self.angle) + '.png')
        plt.clf()

        return vertical_lines, horizontal_lines

    def createLinesCollection(self):
        vertical_lines, horizontal_lines = self.findLines()
        lines_collection = MarkerLinesCollection.MarkerLinesCollection()
        lines_collection.addVerticalLines(vertical_lines)
        lines_collection.addHorizontalLines(horizontal_lines)
        lines_collection.aggregateMarkerLines(self.img_x, self.img_y, self.angle)

        return lines_collection

    def extractCorners(self, visualize=False):
        self.extractMask(visualize=visualize)
        lines_collection = self.createLinesCollection()
        markers, grouped_points_all = lines_collection.getAllMarkersCorners()
        self.marker_front, self.marker_back, self.marker_right, self.marker_left = markers

        if visualize:
            colors = [
                (0, 0, 255),
                (0, 255, 0),
                (255, 255, 0),
                (255, 0, 0),
            ]

            img = self.img.copy()
            for grouped_points in grouped_points_all:
                for points_group, color in zip(grouped_points, colors):
                    for point in points_group:
                        cv2.circle(img, (int(point.x), int(point.y)), radius=10, color=color, thickness=-1)

            plt.imshow(img[:, :, ::-1])
            plt.savefig(utils.dir_results + '/markers_potential_corners' + str(self.angle) + '.png')
            plt.clf()

        return self.marker_front, self.marker_back, self.marker_right, self.marker_left
