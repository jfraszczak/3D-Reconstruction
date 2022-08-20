from ImageInformationExtractor import *


image_information_extractor = ImageInformationExtractor()
image_information_extractor.readImages()
image_information_extractor.extractMarkersCoordinates(visualize=True)
image_information_extractor.visualizeMarkers()

image_information_extractor.extractCrossings()
image_information_extractor.visualizeCrossings()

image_information_extractor.perform3DReconstruction()


