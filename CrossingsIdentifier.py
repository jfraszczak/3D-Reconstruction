import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import math
from CrossingsExtractor import *
from sklearn.linear_model import LinearRegression


class CrossingsIdentifier(CrossingsExtractor):

    def __init__(self, crossings_extractor):
        self.__dict__.update(crossings_extractor.__dict__)
        self.crossing_matrix = [[None] * 20 for _ in range(20)]
        self.crossing_matrix_bottom = [[None] * 20 for _ in range(20)]

    def crossingInArea(self, point, visualise=False):
        coord_x, coord_y = point
        size = 10

        if coord_x + size >= self.img_x or coord_x < 0:
            return
        if coord_y + size >= self.img_y or coord_y - size < 0:
            return

        if visualise:
            cv2.rectangle(self.img, (coord_x, coord_y - size), (coord_x + 10, coord_y + size), (0, 0, 0), 1)

        index = None
        crossing = None
        for x in range(coord_x, coord_x + 10 + 1):
            for y in range(coord_y - size, coord_y + size + 1):
                if self.crossings[y][x] == 1:
                    cv2.circle(self.phase_img, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
                    crossing = (x, y)
                    index = 1
                    return crossing, index
                if self.crossings[y][x] == 2:
                    cv2.circle(self.phase_img, (x, y), radius=3, color=(255, 0, 0), thickness=-1)
                    crossing = (x, y)
                    index = 2

        return crossing, index

    def visualiseCrossings(self):
       for y in range(self.crossings.shape[0]):
           for x in range(self.crossings.shape[1]):
               if self.crossings[y][x] != 0:
                   cv2.circle(self.img, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

    def insertCrossing(self, crossing_matrix, row, column, crossing):
        i = 20 - 1
        while i >= column:
            if crossing_matrix[row][i] is not None and i < 19:
                crossing_matrix[row][i + 1] = crossing_matrix[row][i]
            i -= 1
        crossing_matrix[row][column] = crossing

        return crossing_matrix

    def identifyCrossingsAlongRightGradient(self, point_lower, angle_lower_start, point_upper, angle_upper_start, row_number, column_number, towards_bottom=False):
        best_crossing_candidate = None
        best_index = None

        X = []
        Y = []

        angle_lower = angle_lower_start
        angle_upper = angle_upper_start

        iterations_since_crossing = 0

        if not towards_bottom:
            crossing_matrix = [row[:] for row in self.crossing_matrix]
        else:
            crossing_matrix = [row[:] for row in self.crossing_matrix_bottom]

        for i in range(200):
            point_lower, point_upper = self.nextPointAlongTheRightGradient1(point_lower, point_upper, angle_lower, angle_upper)

            x_lower, y_lower = point_lower
            x_upper, y_upper = point_upper
            angle_lower = self.phase[y_lower][x_lower]
            angle_upper = self.phase[y_upper][x_upper]

            if not self.gradient_min + self.gradient_right_lower <= angle_lower <= self.gradient_max + self.gradient_right_lower:
                angle_lower = angle_lower_start

            if not self.gradient_min <= angle_upper <= self.gradient_max:
                angle_upper = angle_upper_start

            point = (int((point_lower[0] + point_upper[0]) / 2), int((point_lower[1] + point_upper[1]) / 2))

            crossing, index = self.crossingInArea(point)
            cv2.circle(self.phase_img, crossing, radius=6, color=(255, 255, 0), thickness=-1)
            cv2.circle(self.img, crossing, radius=3, color=(255, 0, 0), thickness=-1)

            if crossing is not None:
                if best_crossing_candidate is None:
                    best_crossing_candidate = crossing
                    best_index = index

                else:
                    dist = ((crossing[0] - best_crossing_candidate[0]) ** 2 + (
                            crossing[1] - best_crossing_candidate[1]) ** 2) ** 0.5

                    if not (dist < 20 and best_index == 1 and index == 2):
                        if dist > 13:

                            if row_number <= 20 and column_number <= 15:
                                if crossing_matrix[row_number - 1][column_number - 1] is None:
                                    crossing_matrix[row_number - 1][column_number - 1] = best_crossing_candidate
                                else:
                                    dist = ((crossing_matrix[row_number - 1][column_number - 1][0] - best_crossing_candidate[0]) ** 2 + (
                                            crossing_matrix[row_number - 1][column_number - 1][1] - best_crossing_candidate[1]) ** 2) ** 0.5
                                    if dist > 20:

                                        if not towards_bottom:
                                            if best_crossing_candidate[1] > crossing_matrix[row_number - 1][column_number - 1][1]:
                                                if best_crossing_candidate[0] > crossing_matrix[row_number - 1][column_number - 1][0]:
                                                    crossing_matrix.insert(row_number - 1, [None] * 20)
                                                    crossing_matrix[row_number - 1][column_number - 1] = best_crossing_candidate
                                                else:
                                                    crossing_matrix = self.insertCrossing(crossing_matrix, row_number - 1, column_number - 1, best_crossing_candidate)
                                        else:
                                            if best_crossing_candidate[1] < crossing_matrix[row_number - 1][column_number - 1][1]:
                                                if best_crossing_candidate[0] < crossing_matrix[row_number - 1][column_number - 1][0]:
                                                    crossing_matrix.insert(row_number - 1, [None] * 20)
                                                    crossing_matrix[row_number - 1][column_number - 1] = best_crossing_candidate
                                                else:
                                                    crossing_matrix = self.insertCrossing(crossing_matrix, row_number - 1, column_number - 1, best_crossing_candidate)

                                column_number += 1


                                best_crossing_candidate = crossing
                                best_index = index
                                #cv2.circle(self.img, crossing, radius=3, color=(0, 255, 0), thickness=-1)

                            else:
                                break

            iterations_since_crossing += 1

            cv2.circle(self.phase_img, point_lower, radius=3, color=(255, 255, 255), thickness=-1)
            cv2.circle(self.phase_img, point_upper, radius=3, color=(255, 255, 255), thickness=-1)
            #cv2.circle(self.phase_img, point, radius=3, color=(255, 255, 0), thickness=-1)

            X.append([point_lower[0]])
            Y.append(point_lower[1])

            if len(X) > 5:
                X = X[1:]
                Y = Y[1:]

            if point_lower[0] >= self.img_x * 0.95 or point_lower[0] < self.img_x * 0.05:
                break
            if point_lower[1] >= self.img_y * 0.95 or point_lower[1] < self.img_y * 0.05:
                break
            if point_upper[0] >= self.img_x * 0.95 or point_upper[0] < self.img_x * 0.05:
                break
            if point_upper[1] >= self.img_y * 0.95 or point_upper[1] < self.img_y * 0.05:
                break

        if not towards_bottom:
            self.crossing_matrix = [row[:] for row in crossing_matrix]
        else:
            self.crossing_matrix_bottom = [row[:] for row in crossing_matrix]

    def identifyCrossings(self, image_information):
        point_lower, angle_lower, point_upper, angle_upper = self.findStartingPoint(image_information)
        crossings, crossings_averaged = self.findCrossingsAlongRightGradient(point_lower, angle_lower, point_upper, angle_upper)
        self.identifyCrossingsAlongRightGradient(point_lower, angle_lower, point_upper, angle_upper, 1, 1)
        column_number = 4

        for i, crossing in enumerate(crossings):
            cv2.circle(self.img, crossing[0], radius=3, color=(0, 255, 255), thickness=-1)
            cv2.circle(self.img, crossing[2], radius=3, color=(0, 255, 255), thickness=-1)
            cv2.putText(self.img, str(i), crossing[0],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)

        for crossing in crossings[2:]:
            point_lower, point_upper = self.findStartingPointForLeftGradient(crossing)

            if point_lower == -1 or point_upper == -1:
                continue

            angle_lower = self.phase[point_lower[1]][point_lower[0]]
            angle_upper = self.phase[point_upper[1]][point_upper[0]]

            crossings1, crossings_averaged1 = self.findCrossingsAlongLeftGradient(point_lower, angle_lower, point_upper, angle_upper)
            crossings2, crossings_averaged2 = self.findCrossingsAlongLeftGradientTowardsBottom(point_lower, angle_lower, point_upper, angle_upper)

            crossings_all = [crossings1[:5], crossings2[1:5]]

            if column_number >= 6:
                crossings_all = [crossings1[:10], crossings2[1:10]]

            for crossings_orientation, crossingsA in enumerate(crossings_all):
                towards_bottom = False
                if crossings_orientation == 1:
                    towards_bottom = True

                previous_starting_point_lowerA = None
                cv2.circle(self.img, previous_starting_point_lowerA, radius=6, color=(0, 0, 0), thickness=-1)
                row_number = 1
                for crossingA in crossingsA:
                    cv2.circle(self.phase_img, crossingA[0], radius=6, color=(255, 0, 255), thickness=-1)
                    cv2.circle(self.phase_img, crossingA[2], radius=6, color=(255, 0, 255), thickness=-1)
                    point_lowerA, point_upperA = self.findStartingPointForRightGradient(crossingA)

                    if point_upperA == -1 or point_upperA == -1:
                        continue

                    if previous_starting_point_lowerA is not None:
                        dist = ((previous_starting_point_lowerA[0] - point_lowerA[0]) ** 2 + (previous_starting_point_lowerA[1] - point_lowerA[1]) ** 2) ** 0.5

                        if dist < 10:
                            continue
                    previous_starting_point_lowerA = point_lowerA

                    if column_number < 9:
                        cv2.circle(self.img, point_lowerA, radius=3, color=(255, 0, 255), thickness=-1)

                        angle_lowerA = self.phase[point_lowerA[1]][point_lowerA[0]]
                        angle_upperA = self.phase[point_upperA[1]][point_upperA[0]]
                        crossingsB, crossings_averagedB = self.findCrossingsAlongRightGradient(point_lowerA, angle_lowerA,
                                                                                               point_upperA, angle_upperA)
                        row_number += 1
                        self.identifyCrossingsAlongRightGradient(point_lowerA, angle_lowerA, point_upperA, angle_upperA, row_number, column_number, towards_bottom=towards_bottom)

            column_number += 1

        return self.crossing_matrix, self.crossing_matrix_bottom
