from sklearn.mixture import GaussianMixture
import numpy as np
import cv2


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def show(self):
        print("({}, {})".format(self.x, self.y))

    def getCoordinates(self):
        return [self.x, self.y]


class Line:
    def __init__(self, line):
        self.point1 = Point(line[0], line[1])
        self.point2 = Point(line[2], line[3])

    def getLength(self):
        return ((self.point1.x - self.point2.x) ** 2 + (self.point1.y - self.point2.y) ** 2) ** (1 / 2)


class MarkerCoordinates:
    def __init__(self, corners):
        self.top_left_corner = corners[0]
        self.top_right_corner = corners[1]
        self.bottom_right_corner = corners[2]
        self.bottom_left_corner = corners[3]

    def getCorners(self):
        return [
            self.top_left_corner,
            self.top_right_corner,
            self.bottom_right_corner,
            self.bottom_left_corner
        ]

    def getCornersAsArray(self):
        return np.array([
            self.top_left_corner.getCoordinates(),
            self.top_right_corner.getCoordinates(),
            self.bottom_right_corner.getCoordinates(),
            self.bottom_left_corner.getCoordinates()
        ])


class MarkerLinesCollection:
    def __init__(self):
        self.vertical_lines = []
        self.horizontal_lines = []

        self.marker_front_lines = []
        self.marker_back_lines = []
        self.marker_right_lines = []
        self.marker_left_lines = []

    def addVerticalLines(self, lines):
        for line in lines:
            self.vertical_lines.append(Line(line))

    def addHorizontalLines(self, lines):
        for line in lines:
            self.horizontal_lines.append(Line(line))

    def aggregateMarkerLines(self, img_x, img_y, angle):

        for line in self.vertical_lines + self.horizontal_lines:
            if angle == 0:
                if line.point1.y < img_y / 2 and line.point2.y < img_y / 2:
                    self.marker_front_lines.append(line)
            if angle == 180:
                if line.point1.y < img_y / 2 and line.point2.y < img_y / 2:
                    self.marker_back_lines.append(line)
            if angle == 60:
                if line.point1.y < img_y / 2 and line.point2.y < img_y / 2:
                    if line.point1.x > img_x * 0.5 and line.point2.x > img_x * 0.5:
                        self.marker_front_lines.append(line)
                else:
                    self.marker_left_lines.append(line)
            if angle == 120:
                if line.point1.y < img_y / 2 and line.point2.y < img_y / 2:
                    if line.point1.x < img_x * 0.33 and line.point2.x < img_x * 0.33:
                        self.marker_back_lines.append(line)
                else:
                    if line.point1.x > img_x * 0.33 and line.point2.x > img_x * 0.33:
                        self.marker_left_lines.append(line)

            if angle == 240:
                if line.point1.y < img_y / 2 and line.point2.y < img_y / 2:
                    if line.point1.x > img_x / 2 and line.point2.x > img_x / 2:
                        self.marker_back_lines.append(line)
                else:
                    if line.point1.x < img_x * 0.67 and line.point2.x < img_x * 0.67:
                        self.marker_right_lines.append(line)

            if angle == 300:
                if line.point1.y < img_y / 2 and line.point2.y < img_y / 2:
                    if line.point1.x < img_x * 0.67 and line.point2.x < img_x * 0.67:
                        self.marker_front_lines.append(line)
                else:
                    if line.point1.x > img_x * 0.5 and line.point2.x > img_x * 0.5:
                        self.marker_right_lines.append(line)

    def identifyMarkerCorners(self, lines):
        points = []
        X = None

        for line in lines:
            points.append(line.point1)
            points.append(line.point2)

            if X is None:
                X = np.array([[line.point1.x, line.point1.y]])
            else:
                X = np.append(X, [[line.point1.x, line.point1.y]], axis=0)
            X = np.append(X, [[line.point2.x, line.point2.y]], axis=0)

        gm = GaussianMixture(n_components=4, random_state=0).fit(X)
        labels = gm.predict(X)

        grouped_points = [[], [], [], []]

        for i, point in enumerate(points):
            grouped_points[labels[i]].append(point)

        x_center = sum(map(lambda i: i[0], gm.means_)) / len(gm.means_)
        y_center = sum(map(lambda i: i[1], gm.means_)) / len(gm.means_)

        corners_points = [0, 0, 0, 0]

        for corner, group in zip(gm.means_, grouped_points):
            x = corner[0]
            y = corner[1]

            if x < x_center and y < y_center:
                point_x = min(group, key=lambda p: p.x).x
                point_y = min(group, key=lambda p: p.y).y
                corners_points[0] = Point(point_x, point_y)
            elif x > x_center and y < y_center:
                point_x = max(group, key=lambda p: p.x).x
                point_y = min(group, key=lambda p: p.y).y
                corners_points[1] = Point(point_x, point_y)
            elif x > x_center and y > y_center:
                point_x = max(group, key=lambda p: p.x).x
                point_y = max(group, key=lambda p: p.y).y
                corners_points[2] = Point(point_x, point_y)
            elif x < x_center and y > y_center:
                point_x = min(group, key=lambda p: p.x).x
                point_y = max(group, key=lambda p: p.y).y
                corners_points[3] = Point(point_x, point_y)

        return corners_points, grouped_points

    def getAllMarkersCorners(self):
        marker_front, marker_back, marker_right, marker_left = None, None, None, None
        grouped_points_all = []
        if len(self.marker_front_lines) > 0:
            corners, grouped_points = self.identifyMarkerCorners(self.marker_front_lines)
            marker_front = MarkerCoordinates(corners)
            grouped_points_all.append(grouped_points)
        if len(self.marker_back_lines) > 0:
            corners, grouped_points = self.identifyMarkerCorners(self.marker_back_lines)
            marker_back = MarkerCoordinates(corners)
            grouped_points_all.append(grouped_points)
        if len(self.marker_right_lines) > 0:
            corners, grouped_points = self.identifyMarkerCorners(self.marker_right_lines)
            marker_right = MarkerCoordinates(corners)
            grouped_points_all.append(grouped_points)
        if len(self.marker_left_lines) > 0:
            corners, grouped_points = self.identifyMarkerCorners(self.marker_left_lines)
            marker_left = MarkerCoordinates(corners)
            grouped_points_all.append(grouped_points)

        return [marker_front, marker_back, marker_right, marker_left], grouped_points_all
