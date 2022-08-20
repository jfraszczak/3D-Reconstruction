import cv2
from MarkersCornersExtractor import *
from CrossingsExtractor import *
from CrossingsIdentifier import *
from CameraCalibration import *
import matplotlib.pyplot as plt
import utils
import copy

class ShapeReconstruction:
    def __init__(self, images, angles):
        self.images = images
        self.angles = angles
        self.intrinsic_matrix = None
        self.fundamental_matrix = None
        self.essential_matrix = None
        self.pts1 = []
        self.pts2 = []

    def reset(self):
        self.intrinsic_matrix = None
        self.fundamental_matrix = None
        self.essential_matrix = None
        self.pts1 = []
        self.pts2 = []

    def calibrateCamera(self):
        camera_calibration = CameraCalibration(self.images)
        self.intrinsic_matrix = camera_calibration.calibrateCamera()

    def images_matching(self):
        angle1 = self.angles[1]
        angle2 = self.angles[0]

        if angle1 == 60:
            for i in range(6, 20):
                for j in range(20):
                    try:
                        point2 = self.images[angle2].crossing_matrix[i][j]
                        point1 = self.images[angle1].crossing_matrix[i][j - 2]

                        if point1 is None or point2 is None:
                            continue

                        self.pts1.append([point1[0], point1[1]])
                        self.pts2.append([point2[0], point2[1]])
                    except:
                        pass

        if angle1 == 300:
            for i in range(20):
                for j in range(20):
                    try:
                        point2 = self.images[angle2].crossing_matrix[i][j]
                        point1 = self.images[angle1].crossing_matrix[i + 1][j - 2]

                        if point1 is None or point2 is None:
                            continue

                        self.pts1.append([point1[0], point1[1]])
                        self.pts2.append([point2[0], point2[1]])
                    except:
                        pass

        image1 = self.images[angle1]
        image2 = self.images[angle2]

        marker1 = image1.marker_front
        marker2 = image2.marker_front

        for point in marker1.getCornersAsArray():
            self.pts1.append([point[0], point[1]])

        for point in marker2.getCornersAsArray():
            self.pts2.append([point[0], point[1]])

        self.pts1 = np.int32(self.pts1)
        self.pts2 = np.int32(self.pts2)

    def getCorrespondingPointsFromFile(self):
        self.pts1 = []
        self.pts2 = []
        file = open("CorrespondingPoints.txt", "r")
        lines = file.readlines()
        for line in lines:
            x1, y1, x2, y2 = line.split()
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            self.pts1.append([x1, y1])
            self.pts2.append([x2, y2])
        file.close()

        self.pts1 = np.int32(self.pts1)
        self.pts2 = np.int32(self.pts2)

    def fundamentalMatrixEstimation(self):
        self.fundamental_matrix, _ = cv2.findFundamentalMat(self.pts1, self.pts2, cv2.FM_LMEDS)

    def drawlines(self, img1, img2, lines, pts1, pts2, colors):
        ''' img1 - image on which we draw the epilines for the points in img2
            lines - corresponding epilines '''
        r, c = img1.shape
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        i = 0
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = colors[i]
            i += 1
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
            img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
            img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
        return img1, img2

    def epipolarLinesComputation(self):
        angle1 = self.angles[1]
        angle2 = self.angles[0]

        img1 = self.images[angle1].img
        img2 = self.images[angle2].img

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        colors = []
        for i in range(1000):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            colors.append(color)

        pts1 = self.pts1[:]
        pts2 = self.pts2[:]

        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, self.fundamental_matrix)
        lines1 = lines1.reshape(-1, 3)
        img5, img6 = self.drawlines(img1, img2, lines1, pts1, pts2, colors)
        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, self.fundamental_matrix)
        lines2 = lines2.reshape(-1, 3)
        img3, img4 = self.drawlines(img2, img1, lines2, pts2, pts1, colors)
        plt.subplot(121)
        plt.imshow(img5)
        plt.subplot(122)
        plt.imshow(img3)
        plt.show()

        plt.subplot(121)
        plt.imshow(img5)
        plt.subplot(122)
        plt.imshow(img3)
        plt.savefig(utils.dir_results + '/epipolar_lines' + str(self.angles[1]) + '.png')
        plt.clf()

    def epipolarConstraint(self):
        angle1 = self.angles[1]
        angle2 = self.angles[0]

        img1 = self.images[angle1].img
        img2 = self.images[angle2].img

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        p2 = np.int32([[1005., 902.]])
        p1 = np.int32([[1028., 999.]])

        colors = []
        for i in range(1000):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            colors.append(color)

        lines1 = cv2.computeCorrespondEpilines(p2.reshape(-1, 1, 2), 2, self.fundamental_matrix)
        lines1 = lines1.reshape(-1, 3)
        img5, img6 = self.drawlines(img1, img2, lines1, p1, p2, colors)
        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        lines2 = cv2.computeCorrespondEpilines(p1.reshape(-1, 1, 2), 1, self.fundamental_matrix)
        lines2 = lines2.reshape(-1, 3)
        img3, img4 = self.drawlines(img2, img1, lines2, p2, p1, colors)
        plt.subplot(121)
        plt.imshow(img5)
        plt.subplot(122)
        plt.imshow(img3)
        plt.show()

        plt.subplot(121)
        plt.imshow(img5)
        plt.subplot(122)
        plt.imshow(img3)
        plt.savefig(utils.dir_results + '/postepipolar_matching' + str(self.angles[1]) + '.png')
        plt.clf()

        def crossingInArea(point, img):
            coord_x, coord_y = point
            size = 10

            index = None
            crossing = None
            for x in range(coord_x, coord_x + 10 + 1):
                for y in range(coord_y - size, coord_y + size + 1):
                    if self.images[angle1].crossings[y][x] >= 1:
                        cv2.circle(img, (x, y), radius=4, color=(255, 0, 0), thickness=-1)
                        #
                        # plt.imshow(img1)
                        # plt.show()
                        return


        r, c = img1.shape
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        for y in range(len(self.images[angle1].crossings)):
            for x in range(len(self.images[angle1].crossings[0])):
                if self.images[angle1].crossings[y][x] >= 1:
                    cv2.circle(img1, (x, y), 1, (255, 0, 255), -1)

        r = lines1[0]
        x, y = map(int, [0, -r[2] / r[1]])

        for i in range(500):
            cv2.circle(img1, (x, y), 1, (0, 0, 255), -1)
            x, y = map(int, [x + 3, -(r[2] + r[0] * (x + 3)) / r[1]])
            crossingInArea((x, y), img1)

        plt.imshow(img1)
        plt.show()

        plt.imshow(img1)
        plt.savefig(utils.dir_results + '/postepipolar_matching_consideration' + str(self.angles[1]) + '.png')
        plt.clf()

    def essentialMatrixEstimation(self):
        self.essential_matrix, _ = cv2.findEssentialMat(self.pts1, self.pts2, self.intrinsic_matrix)

    def triangulation(self):
        points, R, t, mask_2 = cv2.recoverPose(self.essential_matrix, self.pts1, self.pts2, self.intrinsic_matrix)
        I = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        Pr_1 = np.dot(self.intrinsic_matrix, I)
        Pr_2 = np.hstack((np.dot(self.intrinsic_matrix, R), np.dot(self.intrinsic_matrix, t)))

        pts1_t = self.pts1.T
        pts2_t = self.pts2.T

        def DLT(P1, P2, point1, point2):

            A = [point1[1] * P1[2, :] - P1[1, :],
                 P1[0, :] - point1[0] * P1[2, :],
                 point2[1] * P2[2, :] - P2[1, :],
                 P2[0, :] - point2[0] * P2[2, :]
                 ]
            A = np.array(A).reshape((4, 4))
            B = A.transpose() @ A
            from scipy import linalg
            U, s, Vh = linalg.svd(B, full_matrices=False)

            return Vh[3, 0:3] / Vh[3, 3]

        p3ds = []
        for uv1, uv2 in zip(self.pts1, self.pts2):
            _p3d = DLT(Pr_1, Pr_2, uv1, uv2)
            p3ds.append(_p3d)
        coordinates = np.array(p3ds)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = coordinates[:-4, 0]
        y = coordinates[:-4, 1]
        z = coordinates[:-4, 2]

        ax.scatter(x, y, z, c='r', marker='o')

        x = np.concatenate((coordinates[-4:, 0], [coordinates[-4, 0]]))
        y = np.concatenate((coordinates[-4:, 1], [coordinates[-4, 1]]))
        z = np.concatenate((coordinates[-4:, 2], [coordinates[-4, 2]]))

        ax.scatter(x, y, z, c='g', marker='o')
        ax.plot(x, y, z, c='g', marker='o')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()

    def reconstructShape(self):
        images = copy.deepcopy(self.images)
        angles = copy.deepcopy(self.angles)

        self.images = dict()
        self.images[angles[0]] = images[angles[0]]
        self.images[angles[1]] = images[angles[1]]
        self.angles = angles[:2]

        self.calibrateCamera()
        self.images_matching()
        self.fundamentalMatrixEstimation()
        self.epipolarLinesComputation()
        self.epipolarConstraint()
        self.essentialMatrixEstimation()
        self.triangulation()

        self.images = dict()
        self.images[angles[0]] = images[angles[0]]
        self.images[angles[2]] = images[angles[2]]
        self.angles = [angles[0], angles[2]]
        self.reset()

        self.calibrateCamera()
        self.images_matching()
        self.fundamentalMatrixEstimation()
        self.epipolarLinesComputation()
        self.epipolarConstraint()
        self.essentialMatrixEstimation()
        self.triangulation()