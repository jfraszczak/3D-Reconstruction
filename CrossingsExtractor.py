import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import math
from sklearn.linear_model import LinearRegression
import utils

class CrossingsExtractor:
    def __init__(self, img, angle):
        self.img = img
        self.angle = angle
        self.img_x = img.shape[1]
        self.img_y = img.shape[0]

        self.img_hsv = None
        self.img_preprocessed = None
        self.pink_mask = None
        self.black_mask = None

        self.phase = None
        self.phase_img = None
        self.crossings = np.zeros((self.img_y, self.img_x), dtype=np.uint8)

        self.gradient_min = 30
        self.gradient_max = 60
        self.gradient_right_upper = 0
        self.gradient_right_lower = 180
        self.gradient_left_upper = 90
        self.gradient_left_lower = 270

    def preprocessImage(self):
        img_hsv = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        self.img_hsv = cv2.GaussianBlur(img_hsv, (5, 5), 0)

        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

        th = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        th = ~th

        kernel = np.ones((3, 3), np.uint8)
        img_erosion = cv2.erode(th, kernel, iterations=1)
        img_dilation = cv2.dilate(img_erosion, kernel, iterations=2)
        self.img_preprocessed = img_dilation

        plt.imshow(img_gray, cmap='gray')
        plt.savefig(utils.dir_results + '/crossings_gray' + str(self.angle) + '.png')
        plt.clf()

        plt.imshow(th, cmap='gray')
        plt.savefig(utils.dir_results + '/crossings_threshold' + str(self.angle) + '.png')
        plt.clf()

        plt.imshow(img_dilation, cmap='gray')
        plt.savefig(utils.dir_results + '/crossings_threshold_dilation' + str(self.angle) + '.png')
        plt.clf()

    def findPinkMask(self):
        img = self.img_hsv.copy()
        lower_bound = np.array([140, 30, 120])
        upper_bound = np.array([180, 200, 190])
        mask = cv2.inRange(img, lower_bound, upper_bound)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=6)

        i = cv2.bitwise_and(self.img, self.img, mask=mask)
        plt.imshow(i[:, :, ::-1])
        plt.savefig(utils.dir_results + '/pink_mask' + str(self.angle) + '.png')
        plt.clf()

        return mask

    def calculateGradients(self):
        I = self.img_preprocessed.astype(np.float64)

        # Derivative x
        Kx = -1 * np.array([[-1.0, 0.0, 1.0]])
        Fx = ndimage.convolve(I, Kx)

        # Derivative y
        Ky = -1 * np.array([[-1.0], [0.0], [1.0]])
        Fy = ndimage.convolve(I, Ky)

        magnitude = np.sqrt(Fx ** 2 + Fy ** 2)

        # Orientation
        phase = cv2.phase(Fx, Fy, angleInDegrees=True)  # theta
        mask_phase = np.zeros((I.shape[0], I.shape[1], 3), dtype=np.uint8)

        mask_phase[(magnitude != 0) & (phase >= self.gradient_min + self.gradient_right_upper) & (
                phase <= self.gradient_max + self.gradient_right_upper)] = np.array([0, 0, 255])  # red
        mask_phase[(magnitude != 0) & (phase >= self.gradient_min + self.gradient_left_upper) & (
                phase <= self.gradient_max + self.gradient_left_upper)] = np.array([0, 255, 255])  # yellow
        mask_phase[(magnitude != 0) & (phase >= self.gradient_min + self.gradient_right_lower) & (
                phase <= self.gradient_max + self.gradient_right_lower)] = np.array([0, 255, 0])  # green
        mask_phase[(magnitude != 0) & (phase >= self.gradient_min + self.gradient_left_lower) & (
                phase <= self.gradient_max + self.gradient_left_lower)] = np.array([255, 0, 0])  # blue

        self.phase = phase
        self.phase_img = mask_phase

        plt.imshow(self.phase_img[:, :, ::-1])
        plt.savefig(utils.dir_results + '/crossings_phase' + str(self.angle) + '.png')
        plt.clf()

    def pointInArea(self, point, range_min, range_max, enlarged=False, reduced=False):
        coord_x, coord_y = point
        size = 7

        if enlarged:
            size = 13

        if reduced:
            size = 3

        if coord_x + size >= self.img_x or coord_x < 0:
            return -1
        if coord_y + size >= self.img_y or coord_y - size < 0:
            return -1

        point = None
        for x in range(coord_x - size, coord_x + size + 1):
            for y in range(coord_y, coord_y + size + 1):
                if range_min <= self.phase[y][x] <= range_max:
                    if point is None:
                        point = (x, y)
            for y in range(coord_y, coord_y - size, -1):
                if range_min <= self.phase[y][x] <= range_max:
                    if point is None:
                        point = (x, y)

        if point is not None:
            return point
        return -1

    def pointInRightArea(self, point, range_min, range_max, enlarged=False, reduced=False):
        coord_x, coord_y = point
        size = 7

        if enlarged:
            size = 10

        if reduced:
            size = 3

        if coord_x + size >= self.img_x or coord_x < 0:
            return -1
        if coord_y + size >= self.img_y or coord_y - size < 0:
            return -1

        # cv2.line(self.phase_img, (coord_x, coord_y - size), (coord_x, coord_y + size), color=(255, 255, 0), thickness=1)
        # cv2.line(self.phase_img, (coord_x, coord_y), (coord_x + size, coord_y), color=(255, 255, 0), thickness=1)
        k = 0
        point = None
        for x in range(coord_x, coord_x + size + 1):
            for y in range(coord_y, coord_y + size + 1):
                if range_min <= self.phase[y][x] <= range_max:
                    if point is None:
                        point = (x, y)
                    k += 1
            for y in range(coord_y, coord_y - size, -1):
                if range_min <= self.phase[y][x] <= range_max:
                    if point is None:
                        point = (x, y)
                    k += 1

        if k > 0:
            return point
        return -1

    def pointInLeftArea(self, point, range_min, range_max, enlarged=False, reduced=False):
        coord_x, coord_y = point
        size = 7

        if enlarged:
            size = 13

        if reduced:
            size = 3

        for x in range(coord_x, coord_x - size, -1):
            for y in range(coord_y, coord_y - size, -1):
                if range_min <= self.phase[y][x] <= range_max:
                    return x, y
            for y in range(coord_y, coord_y + size + 1):
                if range_min <= self.phase[y][x] <= range_max:
                    return x, y
        return -1

    def findStartingPoint(self, image_information, leg=False):
        if image_information.angle == 0:
            corner = image_information.marker_front.top_left_corner
            x, y = corner.x, corner.y
            if leg:
                x, y = (850, 2300)

        if image_information.angle == 60:
            corner = image_information.marker_front.top_left_corner
            x, y = corner.x, corner.y
            if leg:
                corner = image_information.marker_left.bottom_right_corner
                x, y = corner.x, corner.y

        if image_information.angle == 120:
            corner = image_information.marker_back.top_left_corner
            x, y = corner.x, corner.y
            if leg:
                corner = image_information.marker_left.bottom_left_corner
                x, y = corner.x, corner.y

        if image_information.angle == 180:
            corner = image_information.marker_back.top_left_corner
            x, y = corner.x, corner.y

        if image_information.angle == 240:
            corner = image_information.marker_back.top_left_corner
            x, y = corner.x, corner.y
            if leg:
                corner = image_information.marker_right.bottom_right_corner
                x, y = corner.x, corner.y

        if image_information.angle == 300:
            corner = image_information.marker_front.top_right_corner
            x, y = (630, 830)

        img = self.img.copy()
        x_start, y_start = x, y
        found = False
        first_edge = False
        first_point = None
        first_angle = None
        while not found:
            y -= 1
            cv2.circle(img, (x, y), radius=2, color=(0, 255, 255), thickness=-1)
            point = self.pointInRightArea((x, y), 210, 240)
            if point != -1:
                first_edge = True
                first_point = point
                first_angle = self.phase[first_point[1]][first_point[0]]
            point = self.pointInRightArea((x, y), 30, 60)
            if point != -1 and first_edge:
                x, y = point
                found = True
                angle = self.phase[y][x]

        cv2.circle(img, (x_start, y_start), radius=2, color=(255, 255, 0), thickness=-1)
        cv2.circle(img, first_point, radius=2, color=(255, 0, 255), thickness=-1)
        cv2.circle(img, point, radius=2, color=(255, 0, 255), thickness=-1)

        return first_point, first_angle, point, angle



    def nextPointAlongTheRightGradient(self, point_lower, point_upper, angle_lower_start, angle_upper_start, indicate_no_line_detected=False):
        dist_to_move = 3
        x_lower, y_lower = point_lower
        x_upper, y_upper = point_upper
        angle_lower = self.phase[y_lower][x_lower]
        angle_upper = self.phase[y_upper][x_upper]

        point_lower = -1
        point_upper = -1

        if not self.gradient_min + self.gradient_right_lower <= angle_lower <= self.gradient_max + self.gradient_right_lower:
            angle_lower = angle_lower_start

        if not self.gradient_min <= angle_upper <= self.gradient_max:
            angle_upper = angle_upper_start

        x_upper_next = int(x_upper + math.sin(math.radians(angle_upper)) * dist_to_move)
        y_upper_next = int(y_upper - math.cos(math.radians(angle_upper)) * dist_to_move)

        x_lower_next = int(x_lower + math.sin(math.radians(angle_lower - self.gradient_right_lower)) * dist_to_move)
        y_lower_next = int(y_lower - math.cos(math.radians(angle_lower - self.gradient_right_lower)) * dist_to_move)

        point_upper = self.pointInRightArea((x_upper_next, y_upper_next), self.gradient_min, self.gradient_max)

        if point_upper != -1:
            x_orthogonal, y_orthogonal = point_upper

            for i in range(10):
                x_orthogonal = int(x_orthogonal + math.sin(math.radians(90 - angle_upper)) * i)
                y_orthogonal = int(y_orthogonal + math.cos(math.radians(90 - angle_upper)) * i)

                point = self.pointInRightArea((x_orthogonal, y_orthogonal),
                                         self.gradient_min + self.gradient_right_lower,
                                         self.gradient_max + self.gradient_right_lower)

                if point != -1:
                    point_lower = point
                    break

        no_line_detected = False
        if indicate_no_line_detected:
            if point_lower == -1 and point_upper == -1:
                no_line_detected = True

        if point_lower == -1:
            point_lower = self.pointInRightArea((x_lower_next, y_lower_next),
                                     self.gradient_min + self.gradient_right_lower,
                                     self.gradient_max + self.gradient_right_lower, reduced=True)

            if point_lower == -1:
                point_lower = (x_lower_next, y_lower_next)

        if point_upper == -1:
            point_upper = (x_upper_next, y_upper_next)

        if indicate_no_line_detected:
            return point_lower, point_upper, no_line_detected

        return point_lower, point_upper

    def nextPointAlongTheRightGradient1(self, point_lower, point_upper, angle_lower_start, angle_upper_start):
        dist_to_move = 5
        x_lower, y_lower = point_lower
        x_upper, y_upper = point_upper
        angle_lower = self.phase[y_lower][x_lower]
        angle_upper = self.phase[y_upper][x_upper]

        point_lower = -1
        point_upper = -1

        if not self.gradient_min + self.gradient_right_lower <= angle_lower <= self.gradient_max + self.gradient_right_lower:
            angle_lower = angle_lower_start

        if not self.gradient_min <= angle_upper <= self.gradient_max:
            angle_upper = angle_upper_start

        x_upper_next = int(x_upper + math.sin(math.radians(angle_upper)) * dist_to_move)
        y_upper_next = int(y_upper - math.cos(math.radians(angle_upper)) * dist_to_move)

        x_lower_next = int(x_lower + math.sin(math.radians(angle_lower - self.gradient_right_lower)) * dist_to_move)
        y_lower_next = int(y_lower - math.cos(math.radians(angle_lower - self.gradient_right_lower)) * dist_to_move)

        point_lower = self.pointInRightArea((x_lower_next, y_lower_next),
                                 self.gradient_min + self.gradient_right_lower,
                                 self.gradient_max + self.gradient_right_lower, enlarged=True)

        if point_lower == -1:
            point_lower = (x_lower_next, y_lower_next)

        point_upper = self.pointInRightArea((x_upper_next, y_upper_next),
                                 self.gradient_min,
                                 self.gradient_max, enlarged=True)

        if point_upper == -1:
            point_upper = (x_upper_next, y_upper_next)

        return point_lower, point_upper

    def nextPointAlongTheLeftGradient(self, point_lower, point_upper, angle_lower_start, angle_upper_start):
        dist_to_move = 3
        x_lower, y_lower = point_lower
        x_upper, y_upper = point_upper
        angle_lower = self.phase[y_lower][x_lower]
        angle_upper = self.phase[y_upper][x_upper]

        if not self.gradient_min + self.gradient_left_lower <= angle_lower <= self.gradient_max + self.gradient_left_lower:
            angle_lower = angle_lower_start

        if not self.gradient_min + self.gradient_left_upper <= angle_upper <= self.gradient_max + self.gradient_left_upper:
            angle_upper = angle_upper_start

        x_upper_next = int(x_upper - math.sin(math.radians(angle_upper - self.gradient_left_upper)) * dist_to_move)
        y_upper_next = int(y_upper - math.cos(math.radians(angle_upper - self.gradient_left_upper)) * dist_to_move)
        point_upper = self.pointInLeftArea((x_upper_next, y_upper_next), self.gradient_min + self.gradient_left_upper,
                                           self.gradient_max + self.gradient_left_upper)

        if point_upper != -1:
            x_orthogonal, y_orthogonal = point_upper

            for i in range(10):
                x_orthogonal = int(
                    x_orthogonal - math.sin(math.radians(90 - (angle_upper - self.gradient_left_upper))) * i)
                y_orthogonal = int(
                    y_orthogonal + math.cos(math.radians(90 - (angle_upper - self.gradient_left_upper))) * i)

                point = self.pointInArea((x_orthogonal, y_orthogonal),
                                         self.gradient_min + self.gradient_left_lower,
                                         self.gradient_max + self.gradient_left_lower, reduced=True)

                if point != -1:
                    point_lower = point
                    break

        else:
            point_upper = (x_upper_next, y_upper_next)

            x_lower_next = int(x_lower - math.sin(math.radians(angle_lower - self.gradient_left_lower)) * dist_to_move)
            y_lower_next = int(y_lower - math.cos(math.radians(angle_lower - self.gradient_left_lower)) * dist_to_move)
            point_lower = self.pointInLeftArea((x_lower_next, y_lower_next),
                                               self.gradient_min + self.gradient_left_lower,
                                               self.gradient_max + self.gradient_left_lower)
            if point_lower != -1:
                x_orthogonal, y_orthogonal = point_lower

                for i in range(10):
                    x_orthogonal = int(
                        x_orthogonal + math.sin(math.radians(90 - (angle_lower - self.gradient_left_lower))) * i)
                    y_orthogonal = int(
                        y_orthogonal - math.cos(math.radians(90 - (angle_lower - self.gradient_left_lower))) * i)

                    point = self.pointInArea((x_orthogonal, y_orthogonal),
                                             self.gradient_min + self.gradient_left_upper,
                                             self.gradient_max + self.gradient_left_upper, reduced=True)

                    if point != -1:
                        point_lower = point
                        break

            else:
                point_lower = (x_lower_next, y_lower_next)

        return point_lower, point_upper

    def nextPointAlongTheLeftGradientTowardsBottom(self, point_lower, point_upper, angle_lower_start,
                                                   angle_upper_start):
        dist_to_move = 3
        x_lower, y_lower = point_lower
        x_upper, y_upper = point_upper
        angle_lower = self.phase[y_lower][x_lower]
        angle_upper = self.phase[y_upper][x_upper]

        if not self.gradient_min + self.gradient_left_lower <= angle_lower <= self.gradient_max + self.gradient_left_lower:
            angle_lower = angle_lower_start

        if not self.gradient_min + self.gradient_left_upper <= angle_upper <= self.gradient_max + self.gradient_left_upper:
            angle_upper = angle_upper_start

        x_upper_next = int(x_upper + math.sin(math.radians(angle_upper - self.gradient_left_upper)) * dist_to_move)
        y_upper_next = int(y_upper + math.cos(math.radians(angle_upper - self.gradient_left_upper)) * dist_to_move)
        point_upper = self.pointInRightArea((x_upper_next, y_upper_next), self.gradient_min + self.gradient_left_upper,
                                            self.gradient_max + self.gradient_left_upper)

        if point_upper != -1:
            x_orthogonal, y_orthogonal = point_upper

            for i in range(10):
                x_orthogonal = int(
                    x_orthogonal - math.sin(math.radians(90 - (angle_upper - self.gradient_left_upper))) * i)
                y_orthogonal = int(
                    y_orthogonal + math.cos(math.radians(90 - (angle_upper - self.gradient_left_upper))) * i)

                point = self.pointInArea((x_orthogonal, y_orthogonal),
                                         self.gradient_min + self.gradient_left_lower,
                                         self.gradient_max + self.gradient_left_lower, reduced=True)

                if point != -1:
                    point_lower = point
                    break

        else:
            point_upper = (x_upper_next, y_upper_next)

            x_lower_next = int(x_lower + math.sin(math.radians(angle_lower - self.gradient_left_lower)) * dist_to_move)
            y_lower_next = int(y_lower + math.cos(math.radians(angle_lower - self.gradient_left_lower)) * dist_to_move)
            point_lower = self.pointInRightArea((x_lower_next, y_lower_next),
                                                self.gradient_min + self.gradient_left_lower,
                                                self.gradient_max + self.gradient_left_lower)
            if point_lower != -1:
                x_orthogonal, y_orthogonal = point_lower

                for i in range(10):
                    x_orthogonal = int(
                        x_orthogonal + math.sin(math.radians(90 - (angle_lower - self.gradient_left_lower))) * i)
                    y_orthogonal = int(
                        y_orthogonal - math.cos(math.radians(90 - (angle_lower - self.gradient_left_lower))) * i)

                    point = self.pointInArea((x_orthogonal, y_orthogonal),
                                             self.gradient_min + self.gradient_left_upper,
                                             self.gradient_max + self.gradient_left_upper, reduced=True)

                    if point != -1:
                        point_lower = point
                        break

            else:
                point_lower = (x_lower_next, y_lower_next)

        return point_lower, point_upper

    def cornerPointFoundRightGradient(self, point_lower, point_upper, angle_lower_start, angle_upper_start):
        dist = 3
        x_lower, y_lower = point_lower
        x_upper, y_upper = point_upper

        angle_lower = self.phase[y_lower][x_lower]
        angle_upper = self.phase[y_upper][x_upper]

        if not self.gradient_min + self.gradient_right_lower <= angle_lower <= self.gradient_max + self.gradient_right_lower:
            angle_lower = angle_lower_start

        if not self.gradient_min <= angle_upper <= self.gradient_max:
            angle_upper = angle_upper_start

        x_orthogonal = int(x_upper - math.cos(math.radians(90 - angle_upper)) * dist)
        y_orthogonal = int(y_upper - math.sin(math.radians(90 - angle_upper)) * dist)

        for i in range(dist):
            x_to_check = int(x_orthogonal + math.cos(math.radians(angle_upper)) * i)
            y_to_check = int(y_orthogonal - math.sin(math.radians(angle_upper)) * i)

            # cv2.circle(self.phase_img, (x_to_check, y_to_check), radius=1, color=(0, 0, 255), thickness=-1)

            point = self.pointInRightArea((x_to_check, y_to_check), self.gradient_min + self.gradient_left_lower,
                                          self.gradient_max + self.gradient_left_lower, reduced=True)

            if point != -1:
                return True

        x_orthogonal = int(x_lower + math.cos(math.radians(90 - angle_lower)) * dist)
        y_orthogonal = int(y_lower + math.sin(math.radians(90 - angle_lower)) * dist)

        for i in range(dist):
            x_to_check = int(x_orthogonal + math.cos(math.radians(90 - angle_lower)) * i)
            y_to_check = int(y_orthogonal - math.sin(math.radians(90 - angle_lower)) * i)

            # cv2.circle(self.phase_img, (x_to_check, y_to_check), radius=1, color=(0, 255, 0), thickness=-1)

            point = self.pointInRightArea((x_to_check, y_to_check), self.gradient_min + self.gradient_left_lower,
                                          self.gradient_max + self.gradient_left_lower, reduced=True)

            if point != -1:
                return True

        return False

    def cornerPointFoundLeftGradient(self, point_lower, point_upper, angle_lower_start, angle_upper_start):
        dist = 5
        x_lower, y_lower = point_lower
        x_upper, y_upper = point_upper

        angle_lower = self.phase[y_lower][x_lower]
        angle_upper = self.phase[y_upper][x_upper]

        if not self.gradient_min + self.gradient_left_lower <= angle_lower <= self.gradient_max + self.gradient_left_lower:
            angle_lower = angle_lower_start

        if not self.gradient_min + self.gradient_left_upper <= angle_upper <= self.gradient_max + self.gradient_left_upper:
            angle_upper = angle_upper_start

        x_orthogonal = int(x_upper + math.cos(math.radians(90 - (angle_upper - self.gradient_left_upper))) * dist)
        y_orthogonal = int(y_upper - math.sin(math.radians(90 - (angle_upper - self.gradient_left_upper))) * dist)

        for i in range(dist):
            x_to_check = int(x_orthogonal - math.cos(math.radians(angle_upper - self.gradient_left_upper)) * i)
            y_to_check = int(y_orthogonal - math.sin(math.radians(angle_upper - self.gradient_left_upper)) * i)

            # cv2.circle(self.phase_img, (x_to_check, y_to_check), radius=1, color=(0, 0, 255), thickness=-1)

            point = self.pointInLeftArea((x_to_check, y_to_check), self.gradient_min + self.gradient_right_lower,
                                         self.gradient_max + self.gradient_right_lower, reduced=True)

            if point != -1:
                return True

        x_orthogonal = int(x_lower - math.cos(math.radians(90 - (angle_lower - self.gradient_left_lower))) * dist)
        y_orthogonal = int(y_lower + math.sin(math.radians(90 - (angle_lower - self.gradient_left_lower))) * dist)

        for i in range(dist):
            x_to_check = int(x_orthogonal - math.cos(math.radians(90 - (angle_lower - self.gradient_left_lower))) * i)
            y_to_check = int(y_orthogonal - math.sin(math.radians(90 - (angle_lower - self.gradient_left_lower))) * i)

            # cv2.circle(self.phase_img, (x_to_check, y_to_check), radius=1, color=(0, 255, 0), thickness=-1)

            point = self.pointInLeftArea((x_to_check, y_to_check), self.gradient_min + self.gradient_right_lower,
                                         self.gradient_max + self.gradient_right_lower, reduced=True)

            if point != -1:
                return True

        return False

    def cornerPointFoundLeftGradientTowardsBottom(self, point_lower, point_upper, angle_lower_start, angle_upper_start):
        dist = 5
        x_lower, y_lower = point_lower
        x_upper, y_upper = point_upper

        angle_lower = self.phase[y_lower][x_lower]
        angle_upper = self.phase[y_upper][x_upper]

        if not self.gradient_min + self.gradient_left_lower <= angle_lower <= self.gradient_max + self.gradient_left_lower:
            angle_lower = angle_lower_start

        if not self.gradient_min + self.gradient_left_upper <= angle_upper <= self.gradient_max + self.gradient_left_upper:
            angle_upper = angle_upper_start

        x_orthogonal = int(x_upper + math.cos(math.radians(90 - (angle_upper - self.gradient_left_upper))) * dist)
        y_orthogonal = int(y_upper - math.sin(math.radians(90 - (angle_upper - self.gradient_left_upper))) * dist)

        for i in range(dist):
            x_to_check = int(x_orthogonal + math.cos(math.radians(angle_upper - self.gradient_left_upper)) * i)
            y_to_check = int(y_orthogonal + math.sin(math.radians(angle_upper - self.gradient_left_upper)) * i)

            # cv2.circle(self.phase_img, (x_to_check, y_to_check), radius=1, color=(0, 0, 255), thickness=-1)

            point = self.pointInLeftArea((x_to_check, y_to_check), self.gradient_min + self.gradient_right_lower,
                                         self.gradient_max + self.gradient_right_lower, reduced=True)

            if point != -1:
                return True

        x_orthogonal = int(x_lower - math.cos(math.radians(90 - (angle_lower - self.gradient_left_lower))) * dist)
        y_orthogonal = int(y_lower + math.sin(math.radians(90 - (angle_lower - self.gradient_left_lower))) * dist)

        for i in range(dist):
            x_to_check = int(x_orthogonal + math.cos(math.radians(90 - (angle_lower - self.gradient_left_lower))) * i)
            y_to_check = int(y_orthogonal + math.sin(math.radians(90 - (angle_lower - self.gradient_left_lower))) * i)

            # cv2.circle(self.phase_img, (x_to_check, y_to_check), radius=1, color=(0, 255, 0), thickness=-1)

            point = self.pointInLeftArea((x_to_check, y_to_check), self.gradient_min + self.gradient_right_lower,
                                         self.gradient_max + self.gradient_right_lower, reduced=True)

            if point != -1:
                return True

        return False

    def findCrossingsAlongRightGradient(self, point_lower, angle_lower_start, point_upper, angle_upper_start):
        crossings = []
        crossings_averaged = []
        crossingFound = False
        crossing_upper = None
        crossing_lower = None

        for i in range(200):
            point_lower, point_upper = self.nextPointAlongTheRightGradient(point_lower, point_upper, angle_lower_start,
                                                                           angle_upper_start)

            # cv2.circle(self.phase_img, point_upper, radius=2, color=(255, 255, 255), thickness=-1)
            # cv2.circle(self.phase_img, point_lower, radius=2, color=(255, 255, 255), thickness=-1)

            if point_lower[0] >= self.img_x * 0.95 or point_lower[0] < self.img_x * 0.05:
                break
            if point_lower[1] >= self.img_y * 0.95 or point_lower[1] < self.img_y * 0.05:
                break
            if point_upper[0] >= self.img_x * 0.95 or point_upper[0] < self.img_x * 0.05:
                break
            if point_upper[1] >= self.img_y * 0.95 or point_upper[1] < self.img_y * 0.05:
                break

            if self.cornerPointFoundRightGradient(point_lower, point_upper, angle_lower_start, angle_upper_start):
                crossing_lower = point_lower
                crossing_upper = point_upper
                crossingFound = True
            elif crossingFound:
                crossingFound = False

                x_lower, y_lower = point_lower
                x_upper, y_upper = point_upper

                angle_lower = self.phase[y_lower][x_lower]
                angle_upper = self.phase[y_upper][x_upper]

                if not self.gradient_min + self.gradient_right_lower <= angle_lower <= self.gradient_max + self.gradient_right_lower:
                    angle_lower = angle_lower_start

                if not self.gradient_min <= angle_upper <= self.gradient_max:
                    angle_upper = angle_upper_start

                crossings.append((crossing_lower, angle_lower, crossing_upper, angle_upper))

                # cv2.circle(self.phase_img, crossing_upper, radius=3, color=(255, 255, 255), thickness=-1)
                # cv2.circle(self.phase_img, crossing_lower, radius=3, color=(255, 255, 255), thickness=-1)
                x_avg = int((crossing_upper[0] + crossing_lower[0]) / 2)
                y_avg = int((crossing_upper[1] + crossing_lower[1]) / 2)
                crossings_averaged.append((x_avg, y_avg))
                cv2.circle(self.phase_img, (x_avg, y_avg), radius=5, color=(255, 0, 255), thickness=-1)

        return crossings, crossings_averaged

    def findCrossingsAlongLeftGradient(self, point_lower, angle_lower_start, point_upper, angle_upper_start):
        crossings = []
        crossings_averaged = []
        crossingFound = False
        crossing_upper = None
        crossing_lower = None

        for i in range(200):
            point_lower, point_upper = self.nextPointAlongTheLeftGradient(point_lower, point_upper, angle_lower_start,
                                                                          angle_upper_start)

            # cv2.circle(self.phase_img, point_lower, radius=3, color=(255, 0, 255), thickness=-1)
            # cv2.circle(self.phase_img, point_upper, radius=3, color=(255, 0, 255), thickness=-1)

            if point_lower[0] >= self.img_x * 0.95 or point_lower[0] < self.img_x * 0.05:
                break
            if point_lower[1] >= self.img_y * 0.95 or point_lower[1] < self.img_y * 0.05:
                break
            if point_upper[0] >= self.img_x * 0.95 or point_upper[0] < self.img_x * 0.05:
                break
            if point_upper[1] >= self.img_y * 0.95 or point_upper[1] < self.img_y * 0.05:
                break

            # cv2.circle(self.phase_img, point_lower, radius=3, color=(255, 0, 0), thickness=-1)
            # cv2.circle(self.phase_img, point_upper, radius=3, color=(0, 255, 255), thickness=-1)

            if self.cornerPointFoundLeftGradient(point_lower, point_upper, angle_lower_start, angle_upper_start):
                crossing_lower = point_lower
                crossing_upper = point_upper
                crossingFound = True
            elif crossingFound:
                crossingFound = False

                x_lower, y_lower = point_lower
                x_upper, y_upper = point_upper

                angle_lower = self.phase[y_lower][x_lower]
                angle_upper = self.phase[y_upper][x_upper]

                if not self.gradient_min + self.gradient_left_lower <= angle_lower <= self.gradient_max + self.gradient_left_lower:
                    angle_lower = angle_lower_start

                if not self.gradient_min <= angle_upper <= self.gradient_max:
                    angle_upper = angle_upper_start

                crossings.append((crossing_lower, angle_lower, crossing_upper, angle_upper))

                # cv2.circle(self.phase_img, crossing_upper, radius=3, color=(255, 255, 255), thickness=-1)
                # cv2.circle(self.phase_img, crossing_lower, radius=3, color=(255, 255, 255), thickness=-1)

                x_avg = int((crossing_upper[0] + crossing_lower[0]) / 2)
                y_avg = int((crossing_upper[1] + crossing_lower[1]) / 2)
                crossings_averaged.append((x_avg, y_avg))
                cv2.circle(self.phase_img, (x_avg, y_avg), radius=5, color=(255, 255, 0), thickness=-1)


        return crossings, crossings_averaged

    def findCrossingsAlongLeftGradientTowardsBottom(self, point_lower, angle_lower_start, point_upper,
                                                    angle_upper_start):
        crossings = []
        crossings_averaged = []
        crossingFound = False
        crossing_upper = None
        crossing_lower = None

        for i in range(200):
            point_lower, point_upper = self.nextPointAlongTheLeftGradientTowardsBottom(point_lower, point_upper,
                                                                                       angle_lower_start,
                                                                                       angle_upper_start)

            if point_lower[0] >= self.img_x * 0.95 or point_lower[0] < self.img_x * 0.05:
                break
            if point_lower[1] >= self.img_y * 0.95 or point_lower[1] < self.img_y * 0.05:
                break
            if point_upper[0] >= self.img_x * 0.95 or point_upper[0] < self.img_x * 0.05:
                break
            if point_upper[1] >= self.img_y * 0.95 or point_upper[1] < self.img_y * 0.05:
                break

            # cv2.circle(self.phase_img, point_lower, radius=3, color=(255, 0, 0), thickness=-1)
            # cv2.circle(self.phase_img, point_upper, radius=3, color=(0, 255, 255), thickness=-1)

            if self.cornerPointFoundLeftGradientTowardsBottom(point_lower, point_upper, angle_lower_start,
                                                              angle_upper_start):
                crossing_lower = point_lower
                crossing_upper = point_upper
                crossingFound = True
            elif crossingFound:
                crossingFound = False

                x_lower, y_lower = point_lower
                x_upper, y_upper = point_upper

                angle_lower = self.phase[y_lower][x_lower]
                angle_upper = self.phase[y_upper][x_upper]

                if not self.gradient_min + self.gradient_left_lower <= angle_lower <= self.gradient_max + self.gradient_left_lower:
                    angle_lower = angle_lower_start

                if not self.gradient_min <= angle_upper <= self.gradient_max:
                    angle_upper = angle_upper_start

                crossings.append((crossing_lower, angle_lower, crossing_upper, angle_upper))

                # cv2.circle(self.phase_img, crossing_upper, radius=3, color=(255, 255, 255), thickness=-1)
                # cv2.circle(self.phase_img, crossing_lower, radius=3, color=(255, 255, 255), thickness=-1)

                x_avg = int((crossing_upper[0] + crossing_lower[0]) / 2)
                y_avg = int((crossing_upper[1] + crossing_lower[1]) / 2)
                crossings_averaged.append((x_avg, y_avg))
                cv2.circle(self.phase_img, (x_avg, y_avg), radius=5, color=(0, 255, 0), thickness=-1)


        return crossings, crossings_averaged



    def findStartingPointForLeftGradient(self, crossing):
        crossing_lower, angle_lower, crossing_upper, angle_upper = crossing

        x, y = crossing_upper
        angle = angle_upper

        dist_to_move = 0
        lower_corner = -1
        while lower_corner == -1 and dist_to_move < 10:
            lower_corner = self.pointInArea((x, y), self.gradient_min + self.gradient_left_lower,
                                            self.gradient_max + self.gradient_left_lower)

            x = int(x + math.sin(math.radians(angle)) * dist_to_move)
            y = int(y - math.cos(math.radians(angle)) * dist_to_move)
            point_next = self.pointInRightArea((x, y), self.gradient_min, self.gradient_max)

            if point_next != -1:
                x, y = point_next
                angle = self.phase[y][x]
            else:
                angle = angle_upper

            dist_to_move += 1

        upper_corner = -1
        if lower_corner != -1:
            x, y = lower_corner
            angle = self.phase[y][x]
            angle = angle - self.gradient_left_lower

            for dist_to_move in range(10):
                x = int(x + math.sin(math.radians(90 - angle)) * dist_to_move)
                y = int(y - math.cos(math.radians(90 - angle)) * dist_to_move)

                point = self.pointInArea((x, y),
                                         self.gradient_min + self.gradient_left_upper,
                                         self.gradient_max + self.gradient_left_upper, reduced=True)

                if point != -1:
                    upper_corner = point
                    break

        if lower_corner != -1:
            cv2.circle(self.phase_img, lower_corner, radius=3, color=(255, 255, 0), thickness=-1)

        if upper_corner != -1:
            cv2.circle(self.phase_img, upper_corner, radius=3, color=(0, 255, 255), thickness=-1)


        return (lower_corner, upper_corner)

    def findStartingPointForRightGradient(self, crossing):
        crossing_lower, angle_lower, crossing_upper, angle_upper = crossing

        x, y = crossing_upper
        angle = angle_upper

        dist_to_move = 0
        lower_corner = -1
        while lower_corner == -1 and dist_to_move < 10:
            lower_corner = self.pointInRightArea((x, y), self.gradient_min + self.gradient_right_lower,
                                                 self.gradient_max + self.gradient_right_lower, enlarged=True)

            x = int(x - math.sin(math.radians(angle - self.gradient_left_upper)) * dist_to_move)
            y = int(y - math.cos(math.radians(angle - self.gradient_left_upper)) * dist_to_move)
            point_next = self.pointInLeftArea((x, y), self.gradient_min + self.gradient_left_upper,
                                              self.gradient_max + self.gradient_left_upper)

            if point_next != -1:
                x, y = point_next
                angle = self.phase[y][x]
            else:
                angle = angle_upper

            dist_to_move += 1

        upper_corner = -1
        if lower_corner != -1:
            x, y = lower_corner
            angle = self.phase[y][x]
            angle = angle - self.gradient_left_lower

            for dist_to_move in range(10):
                x = int(x - math.sin(math.radians(90 - (angle - self.gradient_left_upper))) * dist_to_move)
                y = int(y - math.cos(math.radians(90 - (angle - self.gradient_left_upper))) * dist_to_move)

                point = self.pointInLeftArea((x, y),
                                              self.gradient_min + self.gradient_right_upper,
                                              self.gradient_max + self.gradient_right_upper, enlarged=True)

                if point != -1:
                    upper_corner = point
                    break

        # if lower_corner != -1:
        #     cv2.circle(self.phase_img, lower_corner, radius=5, color=(0, 255, 0), thickness=-1)
        #
        # if upper_corner != -1:
        #     cv2.circle(self.phase_img, upper_corner, radius=5, color=(0, 0, 255), thickness=-1)

        return (lower_corner, upper_corner)

    def saveCrossings(self, crossings, index=1):
        for crossing in crossings:
            x, y = crossing
            self.crossings[y][x] = index

    def extractCrossings1(self, point_lower, angle_lower, point_upper, angle_upper):
        crossings, crossings_averaged = self.findCrossingsAlongRightGradient(point_lower, angle_lower, point_upper,
                                                                             angle_upper)
        self.saveCrossings(crossings_averaged, index=1)

        for crossing in crossings:
            point_lower, point_upper = self.findStartingPointForLeftGradient(crossing)

            if point_lower == -1 or point_upper == -1:
                continue
            angle_lower = self.phase[point_lower[1]][point_lower[0]]
            angle_upper = self.phase[point_upper[1]][point_upper[0]]

            crossings1, crossings_averaged1 = self.findCrossingsAlongLeftGradient(point_lower, angle_lower, point_upper,
                                                                                  angle_upper)
            crossings2, crossings_averaged2 = self.findCrossingsAlongLeftGradientTowardsBottom(point_lower, angle_lower,
                                                                                               point_upper, angle_upper)

            crossingsA = crossings1 + crossings2
            crossings_averagedA = crossings_averaged1 + crossings_averaged2
            self.saveCrossings(crossings_averagedA, index=2)

            for crossingA in crossingsA:
                point_lowerA, point_upperA = self.findStartingPointForRightGradient(crossingA)

                if point_upperA == -1 or point_upperA == -1:
                    continue

                angle_lowerA = self.phase[point_lowerA[1]][point_lowerA[0]]
                angle_upperA = self.phase[point_upperA[1]][point_upperA[0]]

                crossingsB, crossings_averagedB = self.findCrossingsAlongRightGradient(point_lowerA, angle_lowerA,
                                                                                       point_upperA, angle_upperA)

                self.saveCrossings(crossings_averagedB, index=1)

        plt.imshow(self.phase_img[:, :, ::-1])
        plt.savefig(utils.dir_results + '/crossings_detected' + str(self.angle) + '.png')
        plt.clf()

    def extractCrossings(self, image_information):
        self.preprocessImage()
        self.calculateGradients()

        point_lower, angle_lower, point_upper, angle_upper = self.findStartingPoint(image_information)
        self.extractCrossings1(point_lower, angle_lower, point_upper, angle_upper)
        point_lower, angle_lower, point_upper, angle_upper = self.findStartingPoint(image_information, leg=True)
        self.extractCrossings1(point_lower, angle_lower, point_upper, angle_upper)

    def getCrossings(self):
        return self.crossings
