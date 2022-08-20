from MarkersCornersExtractor import *
from CrossingsIdentifier import *
from CameraCalibration import *
import matplotlib.pyplot as plt
import utils
from ShapeReconstruction import *


class ImageInformation:
    def __init__(self):
        self.img = None
        self.img_x = None
        self.img_y = None
        self.angle = None
        self.path = None

        self.marker_front = None
        self.marker_back = None
        self.marker_right = None
        self.marker_left = None

        self.crossing_matrix = None
        self.crossing_matrix_bottom = None
        self.pink_mask = None
        self.crossings = None

        self.pink_lines1 = set()
        self.pink_lines2 = set()

    def readImage(self, path, angle):
        self.img = cv2.imread(path, cv2.IMREAD_COLOR)
        self.img_x, self.img_y = self.img.shape[1], self.img.shape[0]
        self.angle = angle
        self.path = path

    def getImage(self):
        return self.img

    def setMarkers(self, markers):
        self.marker_front, self.marker_back, self.marker_right, self.marker_left = markers

    def setCrossingMatrix(self, crossing_matrix, crossing_matrix_bottom):
        self.crossing_matrix = crossing_matrix
        self.crossing_matrix_bottom = crossing_matrix_bottom

    def setPinkMask(self, pink_mask):
        self.pink_mask = pink_mask

    def visualizeMarkers(self):
        img = self.img.copy()
        colors = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 255, 0),
            (255, 0, 0),
        ]

        markers = [self.marker_front, self.marker_back, self.marker_right, self.marker_left]

        for marker in markers:
            if marker is not None:
                corners = marker.getCorners()
                for corner, color in zip(corners, colors):
                    cv2.circle(img, (int(corner.x), int(corner.y)), radius=20, color=color, thickness=-1)

        plt.imshow(img[:, :, ::-1])
        plt.savefig(utils.dir_results + '/markers_corners' + str(self.angle) + '.png')
        plt.clf()

    def mergeCrossingMatrices(self):
        def allNone(row):
            for element in row:
                if element is not None:
                    return False
            return True

        new_matrix = []

        for i in range(len(self.crossing_matrix_bottom) - 1, -1, -1):
            if not allNone(self.crossing_matrix_bottom[i]):
                new_matrix.append(self.crossing_matrix_bottom[i])

        for i in range(len(self.crossing_matrix)):
            if not allNone(self.crossing_matrix[i]):
                new_matrix.append(self.crossing_matrix[i])

        self.crossing_matrix = new_matrix[:]

    def visualiseCrossingMatrix(self):
        img_copy = self.img.copy()
        img_copy = cv2.bitwise_and(img_copy, img_copy, mask=self.pink_mask)
        for i in range(len(self.crossing_matrix)):
            for j in range(len(self.crossing_matrix[i])):
                crossing = self.crossing_matrix[i][j]
                if crossing is not None:
                    cv2.circle(img_copy, crossing, radius=3, color=(0, 0, 255), thickness=-1)
                    cv2.putText(img_copy, str(i) + '.' + str(j), crossing,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # for i in range(15):
        #     for j in range(15):
        #         crossing = self.crossing_matrix_bottom[i][j]
        #         if crossing is not None:
        #             cv2.circle(img_copy, crossing, radius=3, color=(0, 0, 255), thickness=-1)
        #             cv2.putText(img_copy, str(-i) + '.' + str(j), crossing,
        #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


        plt.imshow(img_copy[:, :, ::-1])
        plt.show()

        plt.imshow(img_copy[:, :, ::-1])
        plt.savefig(utils.dir_results + '/crossings_identified' + str(self.angle) + '.png')
        plt.clf()

    def findPinkCurves(self):
        n_max = 0
        best_candidate_curve1 = None
        for i in range(len(self.crossing_matrix)):
            n = 0
            for j in range(len(self.crossing_matrix[0])):
                if self.crossing_matrix[i][j] is not None:
                    x, y = self.crossing_matrix[i][j]
                    if self.pink_mask[y][x]:
                        n += 1

            if n > n_max:
                best_candidate_curve1 = i
                n_max = n

        n_max = 0
        best_candidate_curve2 = None
        for j in range(len(self.crossing_matrix[0])):
            n = 0
            for i in range(len(self.crossing_matrix)):
                if self.crossing_matrix[i][j] is not None:
                    x, y = self.crossing_matrix[i][j]
                    if self.pink_mask[y][x]:
                        n += 1

            if n > n_max:
                best_candidate_curve2 = j
                n_max = n

        print(best_candidate_curve1, best_candidate_curve2)


class ImageInformationExtractor:
    def __init__(self):
        self.images = {}
        self.angles = [0, 60, 300]

    def readImages(self):
        directory = "images/"
        images_names = [
            "30 gen 0 gr.jpg",
            "30 gen 60 gr.jpg",
             #"30 gen 120 gr.jpg",
            # "30 gen 180 gr.jpg",
            # "30 gen 240 gr.jpg",
             "30 gen 300 gr.jpg"
        ]

        for images_name, angle in zip(images_names, self.angles):
            path = directory + images_name
            image = ImageInformation()
            image.readImage(path, angle)
            self.images[angle] = image

    def extractMarkersCoordinates(self, visualize=False):
        for angle in self.angles:
            image = self.images[angle]
            img = image.getImage()
            markers_corners_extractor = MarkersCornersExtractor(img, angle)
            markers = markers_corners_extractor.extractCorners(visualize=visualize)
            image.setMarkers(markers)

    def visualizeMarkers(self):
        for angle in self.angles:
            image = self.images[angle]
            image.visualizeMarkers()

    def visualizeCrossings(self):
        for angle in self.angles:
            image = self.images[angle]
            image.visualiseCrossingMatrix()

    def mergeCrossingMatrices(self):
        for angle in self.angles:
            image = self.images[angle]
            image.mergeCrossingMatrices()

    def extractCrossings(self):
        for angle in self.angles:

            image = self.images[angle]
            img = image.getImage().copy()
            crossings_extractor = CrossingsExtractor(img, angle)
            crossings_extractor.extractCrossings(self.images[angle])
            crossings = crossings_extractor.getCrossings()
            self.images[angle].crossings = crossings
            pink_mask = crossings_extractor.findPinkMask()
            image.setPinkMask(pink_mask)

            crossings_identifier = CrossingsIdentifier(crossings_extractor)
            crossing_matrix, crossing_matrix_bottom = crossings_identifier.identifyCrossings(self.images[angle])
            image.setCrossingMatrix(crossing_matrix, crossing_matrix_bottom)

        self.mergeCrossingMatrices()
        self.findPinkCurves()

    def findPinkCurves(self):
        for angle in self.angles:
            image = self.images[angle]
            image.findPinkCurves()

    def perform3DReconstruction(self):
        shape_reconstruction = ShapeReconstruction(self.images, self.angles)
        shape_reconstruction.reconstructShape()
