import cv2


def click_event1(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img1, str(len(pts1) + 1),
                    (x, y), font, 0.5,
                    (0, 0, 255), 1)
        cv2.circle(img1, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.imshow('image1', img1)
        pts1.append([x, y])


def click_event2(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img2, str(len(pts2) + 1),
                    (x, y), font, 0.5,
                    (0, 0, 255), 1)
        cv2.circle(img2, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.imshow('image2', img2)
        pts2.append([x, y])


path1 = "images/30 gen 60 gr.jpg"
img1 = cv2.imread(path1, 1)

path2 = "images/30 gen 0 gr.jpg"
img2 = cv2.imread(path2, 1)

(h1, w1) = img1.shape[:2]
(h2, w2) = img2.shape[:2]

img1 = cv2.resize(img1, (int(w1 / 2), int(h1 / 2)))
img2 = cv2.resize(img2, (int(w2 / 2), int(h2 / 2)))

# displaying the image
cv2.imshow('image1', img1)
cv2.imshow('image2', img2)

pts1 = []
pts2 = []

try:
    file = open("CorrespondingPoints.txt", "r")
    lines = file.readlines()
    i = 1
    for line in lines:
        x1, y1, x2, y2 = line.split()
        x1 = int(int(x1) / 2)
        y1 = int(int(y1) / 2)
        x2 = int(int(x2) / 2)
        y2 = int(int(y2) / 2)
        pts1.append([x1, y1])
        pts2.append([x2, y2])

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img2, str(i),
                    (x2, y2), font, 0.5,
                    (0, 0, 255), 1)
        cv2.circle(img2, (x2, y2), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.imshow('image2', img2)

        cv2.putText(img1, str(i),
                    (x1, y1), font, 0.5,
                    (0, 0, 255), 1)
        cv2.circle(img1, (x1, y1), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.imshow('image1', img1)
        i += 1


    file.close()
except:
    pass

# setting mouse handler for the image
# and calling the click_event() function
cv2.setMouseCallback('image1', click_event1)
cv2.setMouseCallback('image2', click_event2)

# wait for a key to be pressed to exit
cv2.waitKey(0)

# close the window
cv2.destroyAllWindows()

print(pts1)
print(pts2)

file = open("CorrespondingPoints.txt", "w")
for point1, point2 in zip(pts1, pts2):
    x1, y1 = point1
    x2, y2 = point2

    file.write("{} {} {} {}\n".format(x1 * 2, y1 * 2, x2 * 2, y2 * 2))

file.close()

